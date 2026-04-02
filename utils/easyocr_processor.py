import easyocr
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import re


class EasyOCRProcessor:

    def __init__(self):
        self.available = False
        self.error_message = None
        try:
            print("🔄 Loading EasyOCR models...")
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            self.available = True
            print("✅ EasyOCR ready!")
        except Exception as e:
            self.available = False
            self.error_message = f"EasyOCR init failed: {str(e)}"
            print(f"❌ {self.error_message}")

    def is_available(self):
        return self.available

    def get_error_message(self):
        return self.error_message

    def preprocess_image(self, image_path):
        """Preprocess image for better OCR accuracy."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")

        h, w = img.shape[:2]

        # Upscale if too small
        if min(h, w) < 600:
            scale = 1200 / min(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31, C=11
        )

        return binary

    def extract_text_from_image(self, image_path, language='en'):
        """Extract text using EasyOCR."""
        if not self.available:
            return f"EasyOCR Error: {self.error_message}"

        try:
            print(f"🔍 EasyOCR processing: {os.path.basename(image_path)}")
            preprocessed = self.preprocess_image(image_path)

            # *** FIX: DO NOT use paragraph=True — it changes the return format
            #          from (bbox, text, conf) to (bbox, text), breaking unpacking.
            results = self.reader.readtext(
                preprocessed,
                detail=1,
                paragraph=False,       # <-- KEY FIX
                batch_size=4,
                contrast_ths=0.2,
                adjust_contrast=0.8,
                text_threshold=0.6,
                low_text=0.3,
            )

            if not results:
                return "No text detected in the image."

            all_text = []
            high_confidence_text = []

            for item in results:
                # Safe unpacking for both 2-tuple and 3-tuple
                if len(item) == 3:
                    _, text, confidence = item
                elif len(item) == 2:
                    _, text = item
                    confidence = 0.5
                else:
                    continue

                text = text.strip()
                if not text:
                    continue

                all_text.append(text)
                if confidence > 0.3:
                    high_confidence_text.append(text)

            final_text = ' '.join(high_confidence_text) if high_confidence_text else ' '.join(all_text)
            final_text = self.clean_extracted_text(final_text)
            print(f"🎉 EasyOCR done: {len(final_text)} chars")
            return final_text

        except Exception as e:
            error_msg = f"EasyOCR processing error: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

    def clean_extracted_text(self, text):
        if not text:
            return text
        text = text.replace('|', 'I')
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)
        return text.strip()


_easy_ocr_processor = None


def get_easyocr_processor():
    global _easy_ocr_processor
    if _easy_ocr_processor is None:
        _easy_ocr_processor = EasyOCRProcessor()
    return _easy_ocr_processor
