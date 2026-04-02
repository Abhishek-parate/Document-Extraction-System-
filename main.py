from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify, send_file
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from models import db, Document
from utils.advanced_ocr_processor import AdvancedOCRProcessor
from utils.translator import Translator
from utils.pdf_generator import PDFGenerator
from config import Config
import os
import uuid

main = Blueprint('main', __name__)

pdf_generator = PDFGenerator()
translator    = Translator(Config.GROQ_API_KEY) if Config.GROQ_API_KEY else None

# Pass Groq API key so Vision LLM is used as primary OCR method
ocr_processor = AdvancedOCRProcessor(groq_api_key=Config.GROQ_API_KEY)

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@main.route('/')
def index():
    return render_template('index.html')


@main.route('/dashboard')
@login_required
def dashboard():
    documents = Document.query.filter_by(user_id=current_user.id).order_by(
        Document.created_at.desc()
    ).all()
    return render_template('dashboard.html', documents=documents)


@main.route('/extract')
@login_required
def extract():
    return render_template('extract.html', languages=Config.SUPPORTED_LANGUAGES)


@main.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if not ocr_processor.is_available():
        return jsonify({
            'error': 'No OCR service available.',
            'available_methods': ocr_processor.get_available_methods(),
        }), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not (file and allowed_file(file.filename)):
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, JPEG, or PDF.'}), 400

    try:
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        file.save(filepath)
        print(f"File saved: {filepath}")
        print(f"Available OCR methods: {ocr_processor.get_available_methods()}")

        # Extract text — Vision LLM is tried first if Groq key is configured
        result         = ocr_processor.extract_text(filepath)
        extracted_text = result.get('text', '').strip()
        confidence     = result.get('confidence', 0.0)
        method_used    = result.get('method', 'unknown')

        if not extracted_text:
            return jsonify({
                'error': 'Could not extract text from the image.',
                'confidence': confidence,
                'method_used': method_used,
                'suggestions': [
                    'Ensure good, even lighting with no shadows',
                    'Write clearly with dark ink on white/light paper',
                    'Take the photo directly above the document (perpendicular)',
                    'Make sure the image is in focus — avoid blurry photos',
                    'Avoid glossy surfaces that cause glare',
                ]
            }), 400

        document = Document(
            filename=filename,
            original_filename=file.filename,
            file_path=filepath,
            extracted_text=extracted_text,
            user_id=current_user.id
        )
        db.session.add(document)
        db.session.commit()

        print(f"Extracted {len(extracted_text)} chars via {method_used}")

        return jsonify({
            'success':           True,
            'document_id':       document.id,
            'extracted_text':    extracted_text,
            'word_count':        len(extracted_text.split()),
            'confidence':        confidence,
            'method_used':       method_used,
            'text_type':         result.get('text_type', 'handwritten'),
            'available_methods': ocr_processor.get_available_methods(),
        })

    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({
            'error': f'File processing failed: {str(e)}',
            'help': 'Check server logs for details.'
        }), 500


@main.route('/translate', methods=['POST'])
@login_required
def translate():
    data        = request.get_json()
    document_id = data.get('document_id')
    target_lang = data.get('target_language')
    source_lang = data.get('source_language', 'en')

    if not translator:
        return jsonify({'error': 'Translation not available. Configure GROQ_API_KEY.'}), 500
    if not document_id:
        return jsonify({'error': 'No document_id provided'}), 400
    if source_lang == target_lang and source_lang != 'auto':
        return jsonify({'error': 'Source and target languages must differ'}), 400

    try:
        document = Document.query.get_or_404(document_id)
        if document.user_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403

        translated_text          = translator.translate_text(
            document.extracted_text, source_lang, target_lang
        )
        document.translated_text = translated_text
        document.source_language = source_lang
        document.target_language = target_lang
        db.session.commit()
        return jsonify({'success': True, 'translated_text': translated_text})

    except Exception as e:
        print(f"Translation error: {e}")
        return jsonify({'error': f'Translation failed: {str(e)}'}), 500


@main.route('/generate_pdf/<int:document_id>')
@login_required
def generate_pdf(document_id):
    try:
        document     = Document.query.get_or_404(document_id)
        if document.user_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403

        # ✅ NEW (always use original English extracted text only)
        text_content = document.extracted_text
        if not text_content:
            return jsonify({'error': 'No text content for PDF generation'}), 400

        pdf_filename = f"document_{document_id}_{uuid.uuid4().hex[:8]}.pdf"
        pdf_path     = os.path.join(Config.UPLOAD_FOLDER, pdf_filename)

        if pdf_generator.create_pdf(text_content, pdf_path,
                                    f"Document - {document.original_filename}"):
            document.pdf_path = pdf_path
            db.session.commit()
            return send_file(pdf_path, as_attachment=True,
                             download_name=f"{document.original_filename}.pdf")
        return jsonify({'error': 'Failed to generate PDF'}), 500

    except Exception as e:
        print(f"PDF error: {e}")
        return jsonify({'error': f'PDF generation failed: {str(e)}'}), 500


@main.route('/delete_document/<int:document_id>', methods=['POST'])
@login_required
def delete_document(document_id):
    try:
        document = Document.query.get_or_404(document_id)
        if document.user_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403
        for path in [document.file_path, document.pdf_path]:
            if path and os.path.exists(path):
                os.remove(path)
        db.session.delete(document)
        db.session.commit()
        flash('Document deleted.', 'success')
        return redirect(url_for('main.dashboard'))
    except Exception as e:
        flash(f'Delete failed: {str(e)}', 'error')
        return redirect(url_for('main.dashboard'))


@main.route('/reprocess/<int:document_id>', methods=['POST'])
@login_required
def reprocess_document(document_id):
    try:
        document = Document.query.get_or_404(document_id)
        if document.user_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403
        method = request.json.get('method', None)
        result = ocr_processor.extract_text(document.file_path, force_method=method)
        document.extracted_text = result['text']
        db.session.commit()
        return jsonify({
            'success':        True,
            'extracted_text': result['text'],
            'confidence':     result['confidence'],
            'method_used':    result['method'],
        })
    except Exception as e:
        return jsonify({'error': f'Reprocessing failed: {str(e)}'}), 500
