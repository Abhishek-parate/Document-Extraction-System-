import os
import platform
from dotenv import load_dotenv

load_dotenv()

def get_tesseract_path():
    """Auto-detect Tesseract path based on operating system"""
    system = platform.system().lower()
    
    if system == 'windows':
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
            'tesseract.exe'
        ]
    elif system == 'darwin':  # macOS
        possible_paths = [
            '/opt/homebrew/bin/tesseract',
            '/usr/local/bin/tesseract',
            '/usr/bin/tesseract',
            'tesseract'
        ]
    else:  # Linux
        possible_paths = [
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract',
            'tesseract'
        ]
    
    for path in possible_paths:
        if check_tesseract_installation(path):
            print(f"✅ Found Tesseract at: {path}")
            return path
    
    print("⚠️ Tesseract not found (optional - other OCR methods will be used)")
    return None

def check_tesseract_installation(path):
    """Check if Tesseract is installed at the given path"""
    try:
        import subprocess
        result = subprocess.run([path, '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        return result.returncode == 0
    except:
        return False

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///smart_doc_extractor.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Translation API
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
    
    # OCR Configuration
    TESSERACT_CMD = os.environ.get('TESSERACT_CMD') or get_tesseract_path()
    
    # Supported languages for translation
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'hi': 'Hindi',
        'mr': 'Marathi',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'ar': 'Arabic',
        'ru': 'Russian'
    }
    
    @classmethod
    def check_configuration(cls):
        """Check and report configuration status"""
        print("\n" + "="*50)
        print("📋 Document Extraction System Configuration")
        print("="*50)
        print("🆓 All OCR methods are FREE and run locally!")
        print("="*50)
        
        print("\n📊 OCR Methods (in priority order for handwriting):")
        print("  1. TrOCR (Microsoft) - Best for handwriting")
        print("  2. PaddleOCR - Good for both handwritten and printed")
        print("  3. EasyOCR - Good for printed text")
        print("  4. Tesseract - Fast fallback option")
        
        if cls.TESSERACT_CMD:
            print(f"\n✅ Tesseract found at: {cls.TESSERACT_CMD}")
        
        if cls.GROQ_API_KEY:
            print("✅ Groq translation API configured")
        else:
            print("⚠️ Groq API key not found (translation unavailable)")
        
        print("\n💡 First run will download models (~500MB)")
        print("   Models are cached locally after first download")
        print("="*50 + "\n")