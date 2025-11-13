import easyocr
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os

class EasyOCRProcessor:
    def __init__(self):
        """Initialize EasyOCR - first time takes 2-3 minutes to download models"""
        self.available = False
        self.error_message = None
        
        try:
            print("🔄 Loading EasyOCR models (first time will download ~100MB)...")
            # Initialize with English, you can add more languages: ['en', 'hi', 'fr', etc.]
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            self.available = True
            print("✅ EasyOCR ready for high-accuracy text extraction!")
        except Exception as e:
            self.available = False
            self.error_message = f"EasyOCR initialization failed: {str(e)}"
            print(f"❌ EasyOCR Error: {self.error_message}")
    
    def is_available(self):
        return self.available
    
    def get_error_message(self):
        return self.error_message
    
    def enhance_image_for_handwriting(self, image_path):
        """Specialized image enhancement for handwritten text"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Get dimensions
            height, width = img.shape[:2]
            print(f"📐 Original image: {width}x{height}")
            
            # Resize image if too small (EasyOCR works better with larger images)
            if width < 800:
                scale_factor = 1200 / width
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                print(f"📈 Resized to: {new_width}x{new_height}")
            
            # Convert to PIL for enhancement
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Enhance for better OCR
            # Increase contrast significantly for handwriting
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(2.2)
            
            # Increase sharpness for clearer text edges
            enhancer = ImageEnhance.Sharpness(pil_img)
            pil_img = enhancer.enhance(2.8)
            
            # Slightly increase brightness
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(1.15)
            
            # Convert back to OpenCV format
            enhanced_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            return enhanced_img
            
        except Exception as e:
            print(f"❌ Image enhancement error: {str(e)}")
            # Return original image if enhancement fails
            return cv2.imread(image_path)
    
    def extract_text_from_image(self, image_path, language='en'):
        """Extract text using EasyOCR with high accuracy"""
        if not self.available:
            return f"EasyOCR Error: {self.error_message}"
        
        try:
            print(f"🔍 EasyOCR processing: {os.path.basename(image_path)}")
            
            # Enhance image for better OCR
            enhanced_img = self.enhance_image_for_handwriting(image_path)
            
            # Extract text with EasyOCR
            print("🧠 Running deep learning OCR analysis...")
            results = self.reader.readtext(enhanced_img, detail=1)
            
            if not results:
                return "No text detected in the image."
            
            # Process results with confidence filtering
            all_text = []
            high_confidence_text = []
            
            print("📊 OCR Results Analysis:")
            for i, (bbox, text, confidence) in enumerate(results):
                print(f"   {i+1}. '{text}' (confidence: {confidence:.3f})")
                
                # Add all text
                all_text.append(text)
                
                # Only include high-confidence text
                if confidence > 0.4:  # Lower threshold for handwriting
                    high_confidence_text.append(text)
            
            # Choose best result
            if high_confidence_text:
                final_text = ' '.join(high_confidence_text)
                print(f"✅ High-confidence result: {len(final_text)} characters")
            else:
                final_text = ' '.join(all_text)
                print(f"⚠️ Using all detected text: {len(final_text)} characters")
            
            # Clean up text
            final_text = self.clean_extracted_text(final_text)
            
            print(f"🎉 EasyOCR extraction completed: '{final_text[:100]}...'")
            return final_text
            
        except Exception as e:
            error_msg = f"EasyOCR processing error: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def clean_extracted_text(self, text):
        """Clean and improve extracted text"""
        if not text:
            return text
        
        # Basic cleanup
        import re
        
        # Fix common OCR errors for handwriting
        cleaned = text.replace('|', 'I')  # Pipe to I
        cleaned = cleaned.replace('0', 'O')  # Zero to O where appropriate
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single
        cleaned = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', cleaned)  # Fix sentence spacing
        
        return cleaned.strip()

# Global instance
_easy_ocr_processor = None

def get_easyocr_processor():
    """Get or create EasyOCR processor instance"""
    global _easy_ocr_processor
    if _easy_ocr_processor is None:
        _easy_ocr_processor = EasyOCRProcessor()
    return _easy_ocr_processor