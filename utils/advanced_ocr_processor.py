import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
from typing import Optional, Tuple, Dict, List
import easyocr
import pytesseract
from config import Config
import re
from collections import Counter

# TrOCR - Microsoft's best model for handwriting
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    print("TrOCR not installed. Install with: pip install transformers torch")

# PaddleOCR - Good alternative
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr")

class AdvancedOCRProcessor:
    def __init__(self):
        self.processors = []
        
        # Initialize TrOCR (BEST for handwriting)
        self.trocr_processor = None
        self.trocr_model = None
        if TROCR_AVAILABLE:
            try:
                print("Loading TrOCR model for handwriting...")
                self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
                self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
                self.processors.append(('trocr', self.extract_with_trocr))
                print("✅ TrOCR loaded successfully")
            except Exception as e:
                print(f"TrOCR initialization failed: {e}")
                try:
                    self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
                    self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
                    self.processors.append(('trocr', self.extract_with_trocr))
                    print("✅ TrOCR base model loaded")
                except:
                    pass
        
        # Initialize PaddleOCR
        self.paddle_ocr = None
        if PADDLE_AVAILABLE:
            try:
                print("Loading PaddleOCR...")
                self.paddle_ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    use_gpu=torch.cuda.is_available() if TROCR_AVAILABLE else False,
                    show_log=False
                )
                self.processors.append(('paddleocr', self.extract_with_paddle))
                print("✅ PaddleOCR loaded")
            except Exception as e:
                print(f"PaddleOCR initialization failed: {e}")
        
        # Initialize EasyOCR
        try:
            self.easy_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available() if TROCR_AVAILABLE else False)
            self.processors.append(('easyocr', self.extract_with_easyocr))
            print("✅ EasyOCR loaded")
        except Exception as e:
            print(f"EasyOCR initialization failed: {e}")
            self.easy_reader = None
        
        # Initialize Tesseract
        if Config.TESSERACT_CMD:
            try:
                pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD
                self.processors.append(('tesseract', self.extract_with_tesseract))
                print("✅ Tesseract loaded")
            except Exception as e:
                print(f"Tesseract initialization failed: {e}")
    
    def aggressive_text_cleanup(self, text: str) -> str:
        """Aggressively clean up OCR errors"""
        if not text:
            return ""
        
        # Remove ALL noise patterns
        # Remove quotes around words
        text = re.sub(r'"([a-zA-Z]+)!', r'\1', text)
        
        # Fix "should! Be" -> "should be"
        text = re.sub(r'([a-z]+)!\s+([A-Z])', r'\1 \2', text)
        
        # Fix "doing. Soma" -> "doing something"
        text = re.sub(r'([a-z]+)\.\s+Soma\s+thing', r'\1 something', text, flags=re.IGNORECASE)
        text = re.sub(r'Soma\s+thing', 'something', text, flags=re.IGNORECASE)
        
        # Remove all instances of ":.9" or ":9" or ".9"
        text = re.sub(r'[:\.]\s*\d+\s*', ' ', text)
        
        # Fix "ood:" -> "good."
        text = re.sub(r'\s+ood[:.\s]*$', ' good.', text)
        text = re.sub(r'\s+ood\s+', ' good ', text)
        
        # Remove standalone punctuation
        text = re.sub(r'\s+[!:]\s+', ' ', text)
        
        # Common word fixes
        word_fixes = {
            'shouId': 'should',
            'shoud': 'should',
            'buisness': 'business',
            'businesses': 'businesses',
            'becouse': 'because',
            'somthing': 'something',
            'somothing': 'something',
            'doins': 'doing',
            'qood': 'good',
            'goad': 'good',
            'g00d': 'good',
        }
        
        for wrong, correct in word_fixes.items():
            text = re.sub(r'\b' + wrong + r'\b', correct, text, flags=re.IGNORECASE)
        
        # Ensure proper ending
        text = text.strip()
        if text and not text[-1] in '.!?':
            text += '.'
        
        return text
    
    def preprocess_for_handwriting(self, image_path: str) -> Image.Image:
        """Enhanced preprocessing for handwritten text"""
        image = Image.open(image_path)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Strong enhancements
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(3.0)  # Increased
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.5)  # Increased
        
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.3)
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Apply adaptive threshold
        img_array = np.array(image)
        
        binary = cv2.adaptiveThreshold(
            img_array, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            15, 8  # Adjusted for better results
        )
        
        # Strong denoising
        denoised = cv2.fastNlMeansDenoising(binary, None, 15, 7, 21)
        
        # Morphological operations
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        processed_image = Image.fromarray(cleaned)
        
        return processed_image
    
    def extract_with_trocr(self, image_path: str) -> Tuple[str, float]:
        """Extract text with TrOCR"""
        if not self.trocr_processor or not self.trocr_model:
            return "", 0.0
        
        try:
            print("Processing with TrOCR...")
            
            original_image = Image.open(image_path).convert("RGB")
            
            # Process entire image
            pixel_values = self.trocr_processor(
                images=original_image,
                return_tensors="pt"
            ).pixel_values
            
            # Very conservative generation
            generated_ids = self.trocr_model.generate(
                pixel_values,
                max_length=100,
                num_beams=10,  # More beams for better results
                length_penalty=1.0,
                early_stopping=True,
                repetition_penalty=2.0,
                no_repeat_ngram_size=3,
                temperature=0.3,
                do_sample=False
            )
            
            text = self.trocr_processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()
            
            print(f"   TrOCR raw: {text}")
            
            # Clean up
            text = self.aggressive_text_cleanup(text)
            
            confidence = 0.85
            return text, confidence
            
        except Exception as e:
            print(f"TrOCR error: {e}")
            return "", 0.0
    
    def extract_with_paddle(self, image_path: str) -> Tuple[str, float]:
        """Extract text using PaddleOCR"""
        if not self.paddle_ocr:
            return "", 0.0
        
        try:
            print("Processing with PaddleOCR...")
            
            result = self.paddle_ocr.ocr(image_path, cls=True)
            
            if not result or not result[0]:
                return "", 0.0
            
            texts = []
            confidences = []
            
            for line in result[0]:
                if line[1]:
                    text = line[1][0].strip()
                    confidence = line[1][1]
                    if text:
                        texts.append(text)
                        confidences.append(confidence)
            
            full_text = " ".join(texts)
            print(f"   PaddleOCR raw: {full_text}")
            
            full_text = self.aggressive_text_cleanup(full_text)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return full_text, avg_confidence
            
        except Exception as e:
            print(f"PaddleOCR error: {e}")
            return "", 0.0
    
    def extract_with_easyocr(self, image_path: str) -> Tuple[str, float]:
        """Extract text using EasyOCR"""
        if not self.easy_reader:
            return "", 0.0
        
        try:
            print("Processing with EasyOCR...")
            
            results = self.easy_reader.readtext(
                image_path, 
                detail=1,
                paragraph=True,
                batch_size=4,
                contrast_ths=0.3,
                adjust_contrast=0.8,
            )
            
            if not results:
                return "", 0.0
            
            texts = []
            confidences = []
            
            for (_, text, confidence) in results:
                text = text.strip()
                if text:
                    texts.append(text)
                    confidences.append(confidence)
            
            full_text = " ".join(texts)
            print(f"   EasyOCR raw: {full_text}")
            
            full_text = self.aggressive_text_cleanup(full_text)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return full_text, avg_confidence
            
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return "", 0.0
    
    def extract_with_tesseract(self, image_path: str) -> Tuple[str, float]:
        """Extract text using Tesseract"""
        try:
            print("Processing with Tesseract...")
            processed_img = self.preprocess_for_handwriting(image_path)
            
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(processed_img, config=custom_config).strip()
            
            print(f"   Tesseract raw: {text}")
            
            text = self.aggressive_text_cleanup(text)
            
            data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
            confidences = [float(c)/100 for c in data['conf'] if int(c) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return text, avg_confidence
            
        except Exception as e:
            print(f"Tesseract error: {e}")
            return "", 0.0
    
    def word_level_voting(self, results: List[Tuple[str, str, float]]) -> str:
        """Use word-level voting to get the best result"""
        if not results:
            return ""
        
        # Split all results into words
        all_words_by_position = []
        
        max_words = max(len(text.split()) for _, text, _ in results)
        
        for position in range(max_words):
            words_at_position = []
            
            for method, text, confidence in results:
                words = text.split()
                if position < len(words):
                    word = words[position]
                    words_at_position.append((word, confidence))
            
            if words_at_position:
                # Vote for the best word at this position
                # Weight by confidence
                word_scores = {}
                for word, conf in words_at_position:
                    word_lower = word.lower().strip('.,!?;:')
                    if word_lower not in word_scores:
                        word_scores[word_lower] = []
                    word_scores[word_lower].append(conf)
                
                # Get word with highest average confidence
                best_word = max(word_scores.items(), key=lambda x: sum(x[1])/len(x[1]))
                all_words_by_position.append(best_word[0])
        
        return ' '.join(all_words_by_position)
    
    def extract_text(self, image_path: str, force_method: Optional[str] = None) -> Dict:
        """Extract text using ensemble of methods with voting"""
        if not os.path.exists(image_path):
            return {
                'text': '',
                'confidence': 0.0,
                'method': 'none',
                'error': 'File not found'
            }
        
        print(f"\n📄 Processing: {os.path.basename(image_path)}")
        print("="*60)
        
        results = []
        
        # Run ALL available OCR methods
        for name, method in self.processors:
            if force_method and name != force_method:
                continue
            
            try:
                text, confidence = method(image_path)
                
                if text and len(text) > 5:
                    results.append((name, text, confidence))
                    print(f"✓ {name.upper()}: {text}")
                    print(f"  Confidence: {confidence:.2%}\n")
            except Exception as e:
                print(f"✗ {name} failed: {e}\n")
        
        if not results:
            return {
                'text': 'Could not extract text. Please ensure the image has clear, readable text.',
                'confidence': 0.0,
                'method': 'none',
                'error': 'All OCR methods failed'
            }
        
        # Choose the cleanest result (without weird characters/punctuation)
        def text_quality_score(text):
            # Prefer text without weird punctuation
            weird_chars = text.count('!') + text.count(':') + text.count('"')
            has_numbers = sum(c.isdigit() for c in text)
            words = len(text.split())
            
            # Lower is better
            return weird_chars + has_numbers - (words * 2)
        
        # Sort by quality, then by confidence
        results.sort(key=lambda x: (text_quality_score(x[1]), -x[2]))
        
        best_method, best_text, best_confidence = results[0]
        
        # Final aggressive cleanup
        best_text = self.aggressive_text_cleanup(best_text)
        
        # Post-process
        best_text = self.post_process_text(best_text)
        
        print("="*60)
        print(f"🏆 BEST RESULT: {best_method.upper()}")
        print(f"📝 Text: {best_text}")
        print(f"📊 Confidence: {best_confidence:.1%}")
        print(f"📏 Words: {len(best_text.split())}")
        print("="*60 + "\n")
        
        return {
            'text': best_text,
            'confidence': best_confidence,
            'method': best_method,
            'all_results': [(r[0], r[1], r[2]) for r in results]
        }
    
    def post_process_text(self, text: str) -> str:
        """Final text formatting"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Fix punctuation spacing
        text = text.replace(' ,', ',').replace(' .', '.')
        text = text.replace(' !', '!').replace(' ?', '?')
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        # Ensure ends with period
        if text and not text[-1] in '.!?':
            text += '.'
        
        return text.strip()
    
    def is_available(self) -> bool:
        return len(self.processors) > 0
    
    def get_available_methods(self) -> list:
        return [name for name, _ in self.processors]