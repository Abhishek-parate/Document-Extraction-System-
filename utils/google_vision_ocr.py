import os
from google.cloud import vision
from PIL import Image
import io

class GoogleVisionOCR:
    def __init__(self, credentials_path=None):
        """Initialize Google Vision OCR"""
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        try:
            self.client = vision.ImageAnnotatorClient()
            self.available = True
            print("✅ Google Vision API initialized successfully")
        except Exception as e:
            self.available = False
            self.error_message = f"Google Vision API error: {str(e)}"
            print(f"❌ Google Vision API error: {str(e)}")
    
    def is_available(self):
        return self.available
    
    def extract_text_from_image(self, image_path):
        """Extract text using Google Vision API"""
        if not self.available:
            return "Google Vision API not available"
        
        try:
            # Read image file
            with io.open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            # Create Vision API image object
            image = vision.Image(content=content)
            
            # Perform text detection
            response = self.client.text_detection(image=image)
            texts = response.text_annotations
            
            if texts:
                # First annotation contains all detected text
                extracted_text = texts[0].description
                print(f"✅ Google Vision extracted {len(extracted_text)} characters")
                return extracted_text
            else:
                return "No text detected in image"
                
        except Exception as e:
            error_msg = f"Google Vision error: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def extract_with_confidence(self, image_path):
        """Extract text with word-level confidence scores"""
        if not self.available:
            return None, []
        
        try:
            with io.open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            response = self.client.document_text_detection(image=image)
            
            if response.full_text_annotation:
                full_text = response.full_text_annotation.text
                
                # Get word-level confidence
                words_info = []
                for page in response.full_text_annotation.pages:
                    for block in page.blocks:
                        for paragraph in block.paragraphs:
                            for word in paragraph.words:
                                word_text = ''.join([symbol.text for symbol in word.symbols])
                                confidence = word.confidence if hasattr(word, 'confidence') else 0.9
                                words_info.append({
                                    'text': word_text,
                                    'confidence': confidence
                                })
                
                return full_text, words_info
            
            return "No text detected", []
            
        except Exception as e:
            print(f"Google Vision detailed error: {str(e)}")
            return None, []

# Google Vision setup instructions
def setup_google_vision():
    """Instructions for setting up Google Vision API"""
    instructions = """
    📋 Google Vision API Setup:
    
    1. Go to Google Cloud Console: https://console.cloud.google.com/
    2. Create a new project or select existing one
    3. Enable the Vision API
    4. Create a service account key (JSON file)
    5. Set environment variable: GOOGLE_APPLICATION_CREDENTIALS=path/to/key.json
    6. You get $300 free credits (enough for ~100,000 OCR requests)
    
    Add to your .env file:
    GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
    """
    return instructions