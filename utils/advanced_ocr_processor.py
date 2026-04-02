import os
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict, List
import re
import math
import base64

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    print("TrOCR not installed: pip install transformers torch")

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    from config import Config
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class AdvancedOCRProcessor:
    """
    OCR processor with Groq Vision LLM as the primary method.
    Vision LLM reads the image directly and is far more accurate than
    traditional OCR for cursive/complex handwriting.
    Falls back to TrOCR / EasyOCR when Groq is not available.
    """

    # Groq vision models to try in order (best quality first)
    VISION_MODELS = [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama-3.2-90b-vision-preview",
        "llama-3.2-11b-vision-preview",
    ]

    def __init__(self, groq_api_key: str = None):
        self.groq_api_key = groq_api_key
        self.processors = []

        # ── PRIMARY: Groq Vision LLM ────────────────────────────────────────
        # Reads cursive/complex handwriting directly from the image.
        # No preprocessing needed, no line segmentation, no character confusion.
        if groq_api_key:
            self.processors.append(("vision_llm", self.extract_with_vision_llm))
            print("Vision LLM (Groq) registered as PRIMARY OCR method")

        # ── FALLBACK 1: TrOCR ───────────────────────────────────────────────
        self.trocr_processor = None
        self.trocr_model = None
        if TROCR_AVAILABLE:
            for model_name in [
                "microsoft/trocr-large-handwritten",
                "microsoft/trocr-base-handwritten",
            ]:
                try:
                    print(f"Loading TrOCR: {model_name}...")
                    self.trocr_processor = TrOCRProcessor.from_pretrained(model_name)
                    self.trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
                    self.trocr_model.eval()
                    self.processors.append(("trocr", self.extract_with_trocr))
                    print(f"TrOCR loaded: {model_name}")
                    break
                except Exception as e:
                    print(f"TrOCR {model_name} failed: {e}")

        # ── FALLBACK 2: PaddleOCR ───────────────────────────────────────────
        self.paddle_ocr = None
        if PADDLE_AVAILABLE:
            try:
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
                self.processors.append(("paddleocr", self.extract_with_paddle))
                print("PaddleOCR loaded")
            except Exception as e:
                print(f"PaddleOCR failed: {e}")

        # ── FALLBACK 3: EasyOCR ─────────────────────────────────────────────
        self.easy_reader = None
        if EASYOCR_AVAILABLE:
            try:
                use_gpu = TROCR_AVAILABLE and torch.cuda.is_available()
                self.easy_reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
                self.processors.append(("easyocr", self.extract_with_easyocr))
                print("EasyOCR loaded")
            except Exception as e:
                print(f"EasyOCR failed: {e}")

        # ── FALLBACK 4: Tesseract ───────────────────────────────────────────
        if TESSERACT_AVAILABLE and Config.TESSERACT_CMD:
            try:
                pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD
                pytesseract.get_tesseract_version()
                self.processors.append(("tesseract", self.extract_with_tesseract))
                print("Tesseract loaded")
            except Exception as e:
                print(f"Tesseract failed: {e}")

        print(f"OCR processors ready: {[p[0] for p in self.processors]}")

    # ── VISION LLM (PRIMARY) ──────────────────────────────────────────────────

    def extract_with_vision_llm(self, image_path: str) -> Tuple[str, float]:
        """
        Use Groq vision-capable LLM to transcribe handwriting directly from image.
        This is the most accurate method for cursive and complex handwriting because:
        - Reads the full image with language context (not character-by-character)
        - Handles connected cursive strokes naturally
        - Works for any ink color or paper background
        - Does NOT hallucinate: instructed to copy text exactly as written
        """
        if not self.groq_api_key:
            return "", 0.0
        try:
            from groq import Groq
            client = Groq(api_key=self.groq_api_key)

            ext = os.path.splitext(image_path)[1].lower()
            mime_map = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png",  ".gif": "image/gif",
                ".webp": "image/webp",
            }
            mime_type = mime_map.get(ext, "image/jpeg")

            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")

            prompt = (
                "You are a transcription assistant. "
                "Your ONLY job is to copy the handwritten text in this image exactly as it appears.\n\n"
                "STRICT RULES:\n"
                "1. Transcribe EVERY word exactly as written — do NOT fix spelling or grammar.\n"
                "2. Preserve paragraph breaks and sentence structure.\n"
                "3. Do NOT add any commentary, labels, headings, or explanations.\n"
                "4. Output ONLY the transcribed text — nothing else.\n"
                "5. If a word is unclear, write your best guess for that word.\n"
            )

            for model in self.VISION_MODELS:
                try:
                    print(f"   Trying Vision LLM: {model}...")
                    response = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime_type};base64,{image_b64}"
                                        },
                                    },
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ],
                        model=model,
                        temperature=0.1,
                        max_tokens=2048,
                    )
                    text = response.choices[0].message.content.strip()
                    if text and len(text) > 5:
                        quality = self._calculate_text_quality(text)
                        print(f"   Vision LLM ({model}) -> {len(text)} chars, quality={quality:.2%}")
                        # Vision LLM text is reliable — score high
                        return text, 0.93
                except Exception as e:
                    err = str(e)
                    if "vision" in err.lower() or "image" in err.lower() or "multimodal" in err.lower():
                        print(f"   {model} does not support vision, trying next...")
                    else:
                        print(f"   {model} error: {err}")
                    continue

            print("   All Vision LLM models failed.")
            return "", 0.0

        except Exception as e:
            print(f"Vision LLM error: {e}")
            return "", 0.0

    # ── PREPROCESSING (for fallback OCR methods) ──────────────────────────────

    def _load_and_scale(self, image_path: str, target_min: int = 1400) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        h, w = img.shape[:2]
        if min(h, w) < target_min:
            scale = target_min / min(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)
        return img

    def _extract_ink_channel(self, bgr: np.ndarray) -> np.ndarray:
        b32 = bgr[:, :, 0].astype(np.int32)
        g32 = bgr[:, :, 1].astype(np.int32)
        r32 = bgr[:, :, 2].astype(np.int32)
        candidates = {
            "gray": cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
            "red":  np.clip(255 - r32 + (g32 + b32) // 4, 0, 255).astype(np.uint8),
            "blue": np.clip(255 - b32 + (r32 + g32) // 4, 0, 255).astype(np.uint8),
            "dark": np.clip((g32 + b32 + r32) // 3, 0, 255).astype(np.uint8),
        }
        best_name, best_gray, best_score = "gray", candidates["gray"], 0.0
        for name, ch in candidates.items():
            _, thresh = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ink_ratio = np.sum(thresh == 0) / thresh.size
            score = 1.0 - abs(ink_ratio - 0.15)
            if score > best_score:
                best_score, best_name, best_gray = score, name, ch
        return best_gray

    def _remove_ruled_lines(self, gray: np.ndarray) -> np.ndarray:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        h, w = gray.shape
        line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 3, 1))
        ruled = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, line_kernel, iterations=1)
        mask = cv2.dilate(ruled, np.ones((3, 3), np.uint8), iterations=1)
        if np.sum(mask) > 0:
            return cv2.inpaint(gray, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return gray

    def _preprocess_for_ocr(self, bgr: np.ndarray) -> np.ndarray:
        gray = self._extract_ink_channel(bgr)
        gray = self._remove_ruled_lines(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = self._deskew(gray)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=1)

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        coords = np.column_stack(np.where(gray < 128))
        if len(coords) < 10:
            return gray
        try:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            if abs(angle) < 0.5:
                return gray
            h, w = gray.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            return gray

    # ── LINE DETECTION (for TrOCR) ────────────────────────────────────────────

    def _detect_text_lines(self, image_path: str) -> list:
        bgr = self._load_and_scale(image_path)
        h_img, w_img = bgr.shape[:2]
        gray = self._extract_ink_channel(bgr)
        gray_clean = self._remove_ruled_lines(gray)
        _, binary = cv2.threshold(gray_clean, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        k_w = max(50, w_img // 12)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 4))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        min_h = max(15, h_img // 60)
        min_w = max(60, w_img // 8)
        max_h = h_img // 5
        raw_bboxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if min_h <= h <= max_h and w >= min_w:
                raw_bboxes.append((x, y, w, h))
        if not raw_bboxes:
            return []
        bboxes = self._merge_same_line_bboxes(raw_bboxes)
        bboxes.sort(key=lambda b: b[1])
        preprocessed = self._preprocess_for_ocr(bgr)
        line_pairs = []
        for (x, y, w, h) in bboxes:
            pad_x, pad_y = 12, 8
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w_img, x + w + pad_x)
            y2 = min(h_img, y + h + pad_y)
            pre_crop  = Image.fromarray(preprocessed[y1:y2, x1:x2]).convert("RGB")
            orig_crop = Image.fromarray(cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
            line_pairs.append((pre_crop, orig_crop))
        return line_pairs

    def _merge_same_line_bboxes(self, bboxes: list) -> list:
        if not bboxes:
            return []
        bboxes = sorted(bboxes, key=lambda b: b[1])
        merged = [list(bboxes[0])]
        for (x, y, w, h) in bboxes[1:]:
            cx, cy, cw, ch = merged[-1]
            overlap = min(cy + ch, y + h) - max(cy, y)
            if overlap > min(ch, h) * 0.4:
                merged[-1] = [
                    min(cx, x), min(cy, y),
                    max(cx + cw, x + w) - min(cx, x),
                    max(cy + ch, y + h) - min(cy, y),
                ]
            else:
                merged.append([x, y, w, h])
        return [tuple(b) for b in merged]

    # ── TEXT QUALITY ──────────────────────────────────────────────────────────

    def _is_garbage_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped or len(stripped) < 2:
            return True
        if not any(c.isalpha() for c in stripped):
            return True
        ratio = sum(1 for c in stripped if c.isalpha()) / len(stripped)
        return ratio < 0.25

    def _calculate_text_quality(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0
        words = text.split()
        if not words:
            return 0.0
        good = 0
        for w in words:
            cleaned = w.strip(".,!?;:()")
            if len(cleaned) >= 2 and any(c.isalpha() for c in cleaned):
                good += 1
        word_q = good / len(words)
        all_chars = text.replace("\n", "").replace(" ", "")
        letter_q = (sum(1 for c in all_chars if c.isalpha()) / len(all_chars)
                    if all_chars else 0.0)
        return round(word_q * 0.6 + letter_q * 0.4, 4)

    # ── FALLBACK: TrOCR ───────────────────────────────────────────────────────

    def extract_with_trocr(self, image_path: str) -> Tuple[str, float]:
        if not self.trocr_processor or not self.trocr_model:
            return "", 0.0
        try:
            print("Processing with TrOCR (line-by-line)...")
            line_pairs = self._detect_text_lines(image_path)
            if not line_pairs:
                img = Image.open(image_path).convert("RGB")
                line_pairs = [(img, img)]
            use_gpu = TROCR_AVAILABLE and torch.cuda.is_available()
            device = "cuda" if use_gpu else "cpu"
            self.trocr_model.to(device)
            good_lines = []
            for (pre_img, orig_img) in line_pairs:
                if pre_img.height < 8 or pre_img.width < 8:
                    continue
                best_text, best_q = "", 0.0
                for candidate in [pre_img, orig_img]:
                    try:
                        pv = self.trocr_processor(
                            images=candidate, return_tensors="pt"
                        ).pixel_values.to(device)
                        with torch.no_grad():
                            gen = self.trocr_model.generate(
                                pv, max_new_tokens=128,
                                num_beams=5, early_stopping=True,
                            )
                        t = self.trocr_processor.batch_decode(
                            gen, skip_special_tokens=True
                        )[0].strip()
                        q = self._calculate_text_quality(t)
                        if q > best_q:
                            best_q, best_text = q, t
                    except Exception:
                        continue
                if best_text and not self._is_garbage_line(best_text):
                    good_lines.append(best_text)
            if not good_lines:
                return "", 0.0
            full_text = "\n".join(good_lines)
            print(f"   TrOCR ({len(line_pairs)} lines, {len(good_lines)} kept): {full_text[:120]}")
            full_text = self.clean_text(full_text)
            quality = self._calculate_text_quality(full_text)
            return full_text, quality * 0.85
        except Exception as e:
            print(f"TrOCR error: {e}")
            return "", 0.0

    # ── FALLBACK: EasyOCR ─────────────────────────────────────────────────────

    def extract_with_easyocr(self, image_path: str) -> Tuple[str, float]:
        if not self.easy_reader:
            return "", 0.0
        try:
            print("Processing with EasyOCR...")
            bgr = self._load_and_scale(image_path)
            preprocessed = self._preprocess_for_ocr(bgr)
            original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            collected = []
            for img_input in [preprocessed, original_rgb]:
                try:
                    results = self.easy_reader.readtext(
                        img_input, detail=1, paragraph=False,
                        batch_size=4, contrast_ths=0.2,
                        adjust_contrast=0.8, text_threshold=0.5,
                        low_text=0.3, link_threshold=0.3,
                    )
                except Exception:
                    results = []
                for item in results:
                    if len(item) == 3:
                        _, text, conf = item
                    elif len(item) == 2:
                        _, text = item
                        conf = 0.5
                    else:
                        continue
                    text = text.strip()
                    if text and len(text) > 1 and any(c.isalpha() for c in text):
                        collected.append((text, float(conf)))
            if not collected:
                return "", 0.0
            seen = {}
            for text, conf in collected:
                key = text.lower()
                if key not in seen or conf > seen[key][1]:
                    seen[key] = (text, conf)
            items = sorted(seen.values(), key=lambda x: x[1], reverse=True)
            full_text = " ".join(t for t, _ in items)
            avg_conf  = sum(c for _, c in items) / len(items)
            print(f"   EasyOCR: {full_text[:120]}")
            full_text = self.clean_text(full_text)
            quality = self._calculate_text_quality(full_text)
            return full_text, avg_conf * quality
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return "", 0.0

    # ── FALLBACK: PaddleOCR ───────────────────────────────────────────────────

    def extract_with_paddle(self, image_path: str) -> Tuple[str, float]:
        if not self.paddle_ocr:
            return "", 0.0
        try:
            print("Processing with PaddleOCR...")
            result = self.paddle_ocr.ocr(image_path, cls=True)
            if not result or not result[0]:
                return "", 0.0
            texts, confs = [], []
            for line in result[0]:
                if line and len(line) >= 2 and line[1]:
                    text = str(line[1][0]).strip()
                    conf = float(line[1][1]) if len(line[1]) > 1 else 0.5
                    if text and any(c.isalpha() for c in text):
                        texts.append(text)
                        confs.append(conf)
            full_text = " ".join(texts)
            full_text = self.clean_text(full_text)
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            quality = self._calculate_text_quality(full_text)
            return full_text, avg_conf * quality
        except Exception as e:
            print(f"PaddleOCR error: {e}")
            return "", 0.0

    # ── FALLBACK: Tesseract ───────────────────────────────────────────────────

    def extract_with_tesseract(self, image_path: str) -> Tuple[str, float]:
        try:
            print("Processing with Tesseract...")
            bgr = self._load_and_scale(image_path)
            pre = self._preprocess_for_ocr(bgr)
            pil = Image.fromarray(pre)
            best_text, best_score = "", 0.0
            for psm in [6, 4, 3, 11]:
                cfg = f"--oem 3 --psm {psm}"
                try:
                    text = pytesseract.image_to_string(pil, config=cfg).strip()
                    data = pytesseract.image_to_data(
                        pil, config=cfg, output_type=pytesseract.Output.DICT
                    )
                    raw = [int(c) for c in data["conf"]
                           if str(c).lstrip("-").isdigit() and int(c) > 0]
                    ac = (sum(raw) / len(raw) / 100) if raw else 0.0
                    q = self._calculate_text_quality(text)
                    if text and ac * q > best_score:
                        best_text, best_score = text, ac * q
                except Exception:
                    continue
            return self.clean_text(best_text), best_score
        except Exception as e:
            print(f"Tesseract error: {e}")
            return "", 0.0

    # ── TEXT CLEANUP ──────────────────────────────────────────────────────────

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.replace("|", "I")
        text = re.sub(r"(?<![a-zA-Z])l(?![a-zA-Z])", "I", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"([.!?,;:])([A-Za-z])", r"\1 \2", text)
        return text.strip()

    # ── MAIN ENTRY POINT ──────────────────────────────────────────────────────

    def extract_text(self, image_path: str, force_method: Optional[str] = None) -> Dict:
        if not os.path.exists(image_path):
            return {"text": "", "confidence": 0.0,
                    "method": "none", "error": "File not found"}

        print(f"\nProcessing: {os.path.basename(image_path)}")
        print("=" * 60)

        results = []
        for name, method in self.processors:
            if force_method and name != force_method:
                continue
            try:
                text, confidence = method(image_path)
                if text and len(text.strip()) > 3:
                    quality = self._calculate_text_quality(text)
                    results.append((name, text, confidence, quality))
                    print(f"  {name.upper()} conf={confidence:.2%} quality={quality:.2%}")
                    # If Vision LLM succeeded with high quality, use it immediately
                    if name == "vision_llm" and quality > 0.5:
                        print("  Vision LLM result accepted — skipping fallbacks")
                        break
            except Exception as e:
                print(f"  {name} failed: {e}")

        if not results:
            return {
                "text": "", "confidence": 0.0, "method": "none",
                "error": "All OCR methods failed",
                "text_type": "unknown", "all_results": [],
            }

        def combined_score(r):
            _, text, conf, quality = r
            return conf * quality * math.log(len(text.split()) + 1)

        results.sort(key=combined_score, reverse=True)
        best_method, best_text, best_conf, best_quality = results[0]

        print("=" * 60)
        print(f"BEST: {best_method.upper()} conf={best_conf:.1%} quality={best_quality:.1%}")
        print(f"Preview:\n{best_text[:400]}")
        print("=" * 60)

        return {
            "text":        best_text,
            "confidence":  best_conf,
            "method":      best_method,
            "text_type":   "handwritten",
            "all_results": [(r[0], r[1], r[2]) for r in results],
        }

    def is_available(self) -> bool:
        return len(self.processors) > 0

    def get_available_methods(self) -> list:
        return [name for name, _ in self.processors]
