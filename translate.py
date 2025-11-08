#!/usr/bin/env python3
"""
Tamil OCR Terminal Application
Upload images via file path and get OCR results with AI validation and multi-language translation
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
from ocr_tamil.ocr import OCR
from paddleocr import PaddleOCR
import re
from openai import OpenAI
from datetime import datetime

# ---------- Torch fix ----------
import torch.serialization
if not hasattr(torch.serialization, "add_safe_globals"):
    def add_safe_globals(*args, **kwargs):
        pass
    torch.serialization.add_safe_globals = add_safe_globals


# ---------- Color codes for terminal output ----------
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


def print_info(text):
    print(f"{Colors.CYAN}ℹ {text}{Colors.ENDC}")


def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")


# ---------- OpenRouter/Qwen Integration ----------
def validate_tamil_text_with_qwen(extracted_text, api_key):
    """Send extracted Tamil text to Qwen2.5 72B model for validation and correction"""
    if not extracted_text or not extracted_text.strip():
        return extracted_text, "No text to validate"

    if not api_key or api_key.strip() == "":
        return extracted_text, "API key not provided"

    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        prompt = f"""You are an expert in Tamil language. You will receive text that was extracted from an image using OCR (Optical Character Recognition). The text may contain errors or mistakes due to OCR inaccuracies.

Your task:
1. Carefully review the extracted Tamil text
2. Identify and correct any OCR errors, spelling mistakes, or formatting issues
3. Ensure proper Tamil grammar and sentence structure
4. Return ONLY the corrected Tamil text without any explanations, comments, or additional formatting
5. If the text is already correct, return it as is
6. Preserve the meaning and intent of the original text

Extracted Tamil Text:
{extracted_text}

Corrected Tamil Text:"""

        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://tamil-ocr-terminal.app",
                "X-Title": "Tamil OCR Terminal Validator",
            },
            model="qwen/qwen-2.5-72b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=2000
        )

        corrected_text = completion.choices[0].message.content.strip()
        return corrected_text, "Success"

    except Exception as e:
        return extracted_text, f"Validation error: {str(e)}"


def translate_text_with_qwen(text, target_languages, api_key):
    """Translate Tamil text to multiple target languages using Qwen2.5 72B"""
    if not text or not text.strip():
        return {}, "No text to translate"

    if not api_key or api_key.strip() == "":
        return {}, "API key not provided"

    if not target_languages:
        return {}, "No target languages specified"

    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        # Normalize language names
        lang_map = {
            'french': 'French',
            'japanese': 'Japanese',
            'chinese': 'Chinese (Simplified)'
        }

        normalized_langs = [lang_map.get(lang.lower(), lang.capitalize()) for lang in target_languages]
        lang_list = ", ".join(normalized_langs)

        prompt = f"""You are an expert translator specializing in Tamil language. Translate the following Tamil text into {lang_list}.

IMPORTANT: 
- Provide ACTUAL TRANSLATIONS with real meaning in each target language
- DO NOT just transliterate Tamil words into other scripts
- Understand the Tamil text meaning and express it naturally in each language
- Each translation should be understandable by native speakers of that language

Tamil Text:
{text}

Provide translations in this EXACT format (one language per section):

FRENCH:
[Write the actual French translation here with real French words]

JAPANESE:
[Write the actual Japanese translation here with real Japanese words]

CHINESE:
[Write the actual Chinese translation here with real Chinese characters]

Provide ONLY the translations in the format above. No other text."""

        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://tamil-ocr-terminal.app",
                "X-Title": "Tamil OCR Terminal Translator",
            },
            model="qwen/qwen-2.5-72b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=3000
        )

        response_text = completion.choices[0].message.content.strip()

        # Parse the response to extract translations
        translations = {}
        lines = response_text.split('\n')
        current_lang = None
        current_text = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Check if it's a language header
            upper_line = stripped.upper()
            if upper_line in ['FRENCH:', 'JAPANESE:', 'CHINESE:', 'CHINESE (SIMPLIFIED):']:
                # Save previous translation
                if current_lang and current_text:
                    translations[current_lang] = '\n'.join(current_text).strip()
                    current_text = []

                # Set new language (normalize header)
                if 'CHINESE' in upper_line:
                    current_lang = 'CHINESE'
                else:
                    current_lang = stripped.replace(':', '').upper()
            elif current_lang and stripped:
                # Skip if it's another language header
                if not any(lang in stripped.upper() for lang in ['FRENCH:', 'JAPANESE:', 'CHINESE:']):
                    current_text.append(stripped)

        # Add the last translation
        if current_lang and current_text:
            translations[current_lang] = '\n'.join(current_text).strip()

        if not translations:
            return {}, "Failed to parse translations from response"

        return translations, "Success"

    except Exception as e:
        return {}, f"Translation error: {str(e)}"


# ---------- OCR Loaders ----------
def load_ocr_model():
    print_info("Loading ocr_tamil model...")
    ocr_tamil = OCR(detect=True)
    print_success("ocr_tamil model loaded")
    return ocr_tamil


def load_paddle_ocr():
    """Load PaddleOCR model for Tamil with CPU"""
    print_info("Loading PaddleOCR model...")
    paddle_ocr = PaddleOCR(lang='ta', use_angle_cls=True, use_gpu=False)
    print_success("PaddleOCR model loaded")
    return paddle_ocr


# ---------- Text Formatting ----------
def format_as_sentences(text):
    """Convert OCR output into proper sentences"""
    if not text:
        return ""

    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'[।.!?]+', text)

    formatted_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            if sentence and sentence[0].isalpha() and sentence[0].isascii():
                sentence = sentence[0].upper() + sentence[1:]
            formatted_sentences.append(sentence)

    if formatted_sentences:
        result = '. '.join(formatted_sentences)
        if not result.endswith('.'):
            result += '.'
        return result
    return text


# ---------- Confidence Assessment ----------
def assess_confidence(detected_text, image_pil):
    """Assess confidence level based on text characteristics"""
    if not detected_text or len(detected_text.strip()) == 0:
        return "No Text"

    text_length = len(detected_text.strip())
    img_np = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Simple heuristics for confidence
    scores = []

    # Text length score
    if text_length > 100:
        scores.append(100)
    elif text_length > 50:
        scores.append(80)
    elif text_length > 20:
        scores.append(60)
    else:
        scores.append(40)

    # Character quality
    total_chars = len(detected_text)
    alpha_chars = sum(1 for c in detected_text if c.isalpha() or ord(c) > 127)
    if total_chars > 0:
        char_quality = (alpha_chars / total_chars) * 100
        scores.append(char_quality)

    # Image clarity
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var > 500:
        scores.append(100)
    elif laplacian_var > 200:
        scores.append(70)
    else:
        scores.append(40)

    avg_score = sum(scores) / len(scores)

    if avg_score >= 80:
        return "Excellent"
    elif avg_score >= 65:
        return "Good"
    elif avg_score >= 50:
        return "Fair"
    elif avg_score >= 30:
        return "Poor"
    else:
        return "Very Poor"


# ---------- Preprocessing Functions ----------
def preprocess_image_original(image_pil: Image.Image):
    """No preprocessing - use original image as-is"""
    return image_pil.convert("RGB")


def preprocess_image_v1(image_pil: Image.Image):
    """Standard preprocessing for printed text"""
    img = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape
    if max(h, w) > 2000:
        scale = 2000 / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    return Image.fromarray(binary)


def preprocess_image_v2(image_pil: Image.Image):
    """Aggressive preprocessing for handwritten/faint text"""
    img = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape
    if max(h, w) > 2000:
        scale = 2000 / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.fastNlMeansDenoising(gray, None, h=30, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 15, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return Image.fromarray(binary)


def preprocess_image_v3(image_pil: Image.Image):
    """Inverted threshold for dark backgrounds"""
    img = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape
    if max(h, w) > 2000:
        scale = 2000 / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.fastNlMeansDenoising(gray, None, h=20, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return Image.fromarray(binary)


def preprocess_image_v4(image_pil: Image.Image):
    """Otsu's thresholding for clear contrast"""
    img = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape
    if max(h, w) > 2000:
        scale = 2000 / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return Image.fromarray(binary)


# ---------- PaddleOCR Method ----------
def run_paddle_ocr(paddle_ocr, image_pil):
    """Run PaddleOCR on the image and return extracted text as string"""
    try:
        img_np = np.array(image_pil.convert("RGB"))
        result = paddle_ocr.ocr(img_np, cls=True)

        text_lines = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2 and line[1]:
                    text_lines.append(str(line[1][0]))

        combined_text = " ".join(text_lines)
        return combined_text.strip()

    except Exception as e:
        return f"Error: {str(e)}"


# ---------- OCR Processing ----------
def run_tamil_ocr_multi(ocr_tamil, paddle_ocr, img_pil, preprocessing_methods,
                        use_paddle, output_dir, base_filename):
    """Try multiple preprocessing methods and save all results"""
    all_results = {}
    original_text = ""
    original_confidence = ""

    for method_name, preprocess_func in preprocessing_methods.items():
        try:
            print_info(f"Processing with {method_name} preprocessing...")
            processed = preprocess_func(img_pil)

            # Save preprocessed image
            preprocessed_filename = f"{base_filename}_{method_name.lower().replace(' ', '_')}.png"
            preprocessed_path = os.path.join(output_dir, preprocessed_filename)
            processed.save(preprocessed_path)
            print_success(f"Saved preprocessed image: {preprocessed_path}")

            if use_paddle:
                text = run_paddle_ocr(paddle_ocr, processed)
            else:
                img_np = np.array(processed.convert("RGB"))
                text = ocr_tamil.predict(img_np)

                if isinstance(text, list):
                    text = " ".join([str(x).strip() for x in text if str(x).strip()])
                elif not isinstance(text, str):
                    text = str(text)

            text = text.strip()
            confidence = assess_confidence(text, processed)

            all_results[method_name] = {
                'text': text,
                'formatted_text': format_as_sentences(text),
                'confidence': confidence,
                'preprocessed_path': preprocessed_path
            }

            print(f"  {Colors.CYAN}Confidence: {confidence}{Colors.ENDC}")

            # Store original method results
            if method_name == "Original":
                original_text = text
                original_confidence = confidence

        except Exception as e:
            print_error(f"Failed with {method_name}: {str(e)}")
            all_results[method_name] = {
                'text': f"Error: {str(e)}",
                'formatted_text': "",
                'confidence': "Failed",
                'preprocessed_path': None
            }

    return original_text, original_confidence, all_results


# ---------- Main Processing Function ----------
def process_image(image_path, output_dir, ocr_engine, preprocessing_mode,
                 api_key, enable_ai, enable_translation, translation_languages,
                 show_details, ocr_tamil, paddle_ocr):
    """Process a single image and return results"""

    print_header(f"Processing: {os.path.basename(image_path)}")

    # Load image
    try:
        img = Image.open(image_path).convert("RGB")
        print_success(f"Image loaded: {img.size[0]}x{img.size[1]} pixels")
    except Exception as e:
        print_error(f"Failed to load image: {str(e)}")
        return

    # Create output directory for this image
    base_filename = Path(image_path).stem
    img_output_dir = os.path.join(output_dir, base_filename)
    os.makedirs(img_output_dir, exist_ok=True)

    use_paddle = (ocr_engine.lower() == "paddleocr")

    # Define preprocessing methods
    methods = {
        "Original": preprocess_image_original,
        "Standard": preprocess_image_v1,
        "Aggressive": preprocess_image_v2,
        "Inverted": preprocess_image_v3,
        "Otsu": preprocess_image_v4
    }

    if preprocessing_mode.lower() == "auto":
        print_info("Running all preprocessing methods...")
        original_text, original_confidence, all_attempts = run_tamil_ocr_multi(
            ocr_tamil, paddle_ocr, img, methods, use_paddle, img_output_dir, base_filename
        )

        # If original method didn't yield text, try to pick the best attempt (highest confidence heuristic)
        if not original_text and all_attempts:
            # choose the attempt with highest confidence string mapped to numeric value if possible
            best_text = ""
            best_conf_val = -1
            best_method_name = None
            for method, info in all_attempts.items():
                conf_label = info.get('confidence', '')
                # map labels to numeric heuristic
                label_map = {'Excellent': 90, 'Good': 75, 'Fair': 55, 'Poor': 35, 'Very Poor': 15, 'No Text': 0, 'Failed': -1}
                val = label_map.get(conf_label, 50)
                if val > best_conf_val and info.get('text') and not info.get('text', '').startswith("Error"):
                    best_conf_val = val
                    best_text = info.get('text', '')
                    best_method_name = method

            if best_text:
                original_text = best_text
                original_confidence = next((all_attempts[m]['confidence'] for m in all_attempts if m == best_method_name), original_confidence)

        if original_text:
            formatted_text = format_as_sentences(original_text)

            # Display metrics
            print(f"\n{Colors.BOLD}OCR Results (Selected Preprocessing):{Colors.ENDC}")
            print(f"  Confidence: {Colors.GREEN}{original_confidence}{Colors.ENDC}")
            print(f"  Characters: {len(original_text)}")

            # AI Validation
            corrected_text = formatted_text
            validation_status = "Skipped"

            if enable_ai and api_key:
                print(f"\n{Colors.CYAN}Running AI validation with Qwen2.5 72B...{Colors.ENDC}")
                corrected_text, validation_status = validate_tamil_text_with_qwen(
                    formatted_text, api_key
                )

                if validation_status == "Success":
                    print_success("AI validation completed")

                    print(f"\n{Colors.BOLD}{'─'*60}{Colors.ENDC}")
                    print(f"{Colors.YELLOW}OCR Extracted Text:{Colors.ENDC}")
                    print(f"{Colors.BOLD}{'─'*60}{Colors.ENDC}")
                    print(formatted_text)

                    print(f"\n{Colors.BOLD}{'─'*60}{Colors.ENDC}")
                    print(f"{Colors.GREEN}AI-Corrected Text:{Colors.ENDC}")
                    print(f"{Colors.BOLD}{'─'*60}{Colors.ENDC}")
                    print(corrected_text)

                    final_text = corrected_text
                else:
                    print_warning(f"AI validation failed: {validation_status}")
                    print(f"\n{Colors.BOLD}{'─'*60}{Colors.ENDC}")
                    print(f"{Colors.YELLOW}Extracted Text (OCR only):{Colors.ENDC}")
                    print(f"{Colors.BOLD}{'─'*60}{Colors.ENDC}")
                    print(formatted_text)
                    final_text = formatted_text
            else:
                if not api_key:
                    print_info("API key not provided - skipping AI validation")
                print(f"\n{Colors.BOLD}{'─'*60}{Colors.ENDC}")
                print(f"{Colors.YELLOW}Extracted Text:{Colors.ENDC}")
                print(f"{Colors.BOLD}{'─'*60}{Colors.ENDC}")
                print(formatted_text)
                final_text = formatted_text

            # Translation
            translations = {}
            translation_status = "Skipped"

            if enable_translation and api_key and translation_languages:
                print(f"\n{Colors.CYAN}{'='*60}{Colors.ENDC}")
                print(f"{Colors.CYAN}{Colors.BOLD}Translating text to {len(translation_languages)} language(s)...{Colors.ENDC}")
                print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")

                translations, translation_status = translate_text_with_qwen(
                    final_text, translation_languages, api_key
                )

                if translation_status == "Success" and translations:
                    print_success(f"Translation completed - {len(translations)} language(s) translated\n")

                    # Display each translation once in order
                    lang_order = ['FRENCH', 'JAPANESE', 'CHINESE']
                    for lang in lang_order:
                        if lang in translations:
                            trans_text = translations[lang]
                            print(f"{Colors.BOLD}{'─'*60}{Colors.ENDC}")
                            print(f"{Colors.BLUE}{Colors.BOLD}▶ {lang} TRANSLATION:{Colors.ENDC}")
                            print(f"{Colors.BOLD}{'─'*60}{Colors.ENDC}")
                            print(f"{trans_text}")
                            print(f"{Colors.BOLD}{'─'*60}{Colors.ENDC}\n")
                else:
                    print_warning(f"Translation failed: {translation_status}")
                    if translations:
                        print(f"{Colors.YELLOW}Partial translations available:{Colors.ENDC}")
                        for lang in translations.keys():
                            print(f"  - {lang}")

            # Save results to file - use original_* variables
            result_file = os.path.join(img_output_dir, f"{base_filename}_result.txt")
            try:
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write(f"Image: {os.path.basename(image_path)}\n")
                    f.write(f"OCR Engine: {ocr_engine}\n")
                    f.write(f"Preprocessing: Auto (selected best)\n")
                    f.write(f"Confidence: {original_confidence}\n")
                    f.write(f"Characters: {len(original_text)}\n")
                    f.write(f"\n{'='*60}\n")
                    f.write("OCR Extracted Text:\n")
                    f.write(f"{'='*60}\n\n")
                    f.write(formatted_text)

                    if enable_ai and api_key and validation_status == "Success":
                        f.write(f"\n\n{'='*60}\n")
                        f.write("AI-Corrected Text:\n")
                        f.write(f"{'='*60}\n\n")
                        f.write(corrected_text)

                    if enable_translation and translations:
                        f.write(f"\n\n{'='*60}\n")
                        f.write("TRANSLATIONS\n")
                        f.write(f"{'='*60}\n")

                        for lang, trans_text in translations.items():
                            f.write(f"\n{lang}:\n")
                            f.write(f"{'-'*60}\n")
                            f.write(trans_text)
                            f.write("\n")
                print_success(f"\nResults saved to: {result_file}")
            except Exception as e:
                print_error(f"Failed to save results: {e}")

            if show_details:
                print(f"\n{Colors.BOLD}All Preprocessing Results:{Colors.ENDC}")
                for method, result in all_attempts.items():
                    print(f"\n  {Colors.CYAN}{method}:{Colors.ENDC}")
                    print(f"    Confidence: {result['confidence']}")
                    print(f"    Preprocessed: {result['preprocessed_path']}")
                    if result['text'] and not result['text'].startswith("Error"):
                        preview = result['text'][:100] + "..." if len(result['text']) > 100 else result['text']
                        print(f"    Text: {preview}")
        else:
            print_error("Could not recognize any text. Try 'auto' mode with clearer images or different preprocessing.")

    else:
        # Single preprocessing method
        method_map = {
            "original": "Original",
            "standard": "Standard",
            "aggressive": "Aggressive",
            "inverted": "Inverted",
            "otsu": "Otsu"
        }
        method_name = method_map.get(preprocessing_mode.lower(), "Standard")
        preprocess_func = methods[method_name]

        print_info(f"Using {method_name} preprocessing...")
        processed = preprocess_func(img)

        # Save preprocessed image
        preprocessed_filename = f"{base_filename}_{method_name.lower()}.png"
        preprocessed_path = os.path.join(img_output_dir, preprocessed_filename)
        processed.save(preprocessed_path)
        print_success(f"Saved preprocessed image: {preprocessed_path}")

        # Run OCR
        if use_paddle:
            text = run_paddle_ocr(paddle_ocr, processed)
        else:
            img_np = np.array(processed.convert("RGB"))
            text = ocr_tamil.predict(img_np)

            if isinstance(text, list):
                text = " ".join([str(x) for x in text])

        raw_text = text.strip()
        formatted_text = format_as_sentences(raw_text)

        if raw_text:
            confidence = assess_confidence(raw_text, processed)

            # Display metrics
            print(f"\n{Colors.BOLD}OCR Results:{Colors.ENDC}")
            print(f"  Confidence: {Colors.GREEN}{confidence}{Colors.ENDC}")
            print(f"  Characters: {len(raw_text)}")

            # AI Validation
            corrected_text = formatted_text
            validation_status = "Skipped"

            if enable_ai and api_key:
                print(f"\n{Colors.CYAN}Running AI validation with Qwen2.5 72B...{Colors.ENDC}")
                corrected_text, validation_status = validate_tamil_text_with_qwen(
                    formatted_text, api_key
                )

                if validation_status == "Success":
                    print_success("AI validation completed")

                    print(f"\n{Colors.BOLD}{'─'*60}{Colors.ENDC}")
                    print(f"{Colors.YELLOW}OCR Extracted Text:{Colors.ENDC}")
                    print(f"{Colors.BOLD}{'─'*60}{Colors.ENDC}")
                    print(formatted_text)

                    print(f"\n{Colors.BOLD}{'─'*60}{Colors.ENDC}")
                    print(f"{Colors.GREEN}AI-Corrected Text:{Colors.ENDC}")
                    print(f"{Colors.BOLD}{'─'*60}{Colors.ENDC}")
                    print(corrected_text)

                    final_text = corrected_text
                else:
                    print_warning(f"AI validation failed: {validation_status}")
                    print(f"\n{Colors.BOLD}{'─'*60}{Colors.ENDC}")
                    print(f"{Colors.YELLOW}Extracted Text (OCR only):{Colors.ENDC}")
                    print(f"{Colors.BOLD}{'─'*60}{Colors.ENDC}")
                    print(formatted_text)
                    final_text = formatted_text
            else:
                if not api_key:
                    print_info("API key not provided - skipping AI validation")
                final_text = formatted_text

            # Translation
            translations = {}
            translation_status = "Skipped"

            if enable_translation and api_key and translation_languages:
                print(f"\n{Colors.CYAN}{'='*60}{Colors.ENDC}")
                print(f"{Colors.CYAN}{Colors.BOLD}Translating text to {len(translation_languages)} language(s)...{Colors.ENDC}")
                print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")

                translations, translation_status = translate_text_with_qwen(
                    final_text, translation_languages, api_key
                )

                if translation_status == "Success" and translations:
                    print_success(f"Translation completed - {len(translations)} language(s) translated\n")

                    lang_order = ['FRENCH', 'JAPANESE', 'CHINESE']
                    for lang in lang_order:
                        if lang in translations:
                            trans_text = translations[lang]
                            print(f"{Colors.BOLD}{'─'*60}{Colors.ENDC}")
                            print(f"{Colors.BLUE}{Colors.BOLD}▶ {lang} TRANSLATION:{Colors.ENDC}")
                            print(f"{Colors.BOLD}{'─'*60}{Colors.ENDC}")
                            print(f"{trans_text}")
                            print(f"{Colors.BOLD}{'─'*60}{Colors.ENDC}\n")
                else:
                    print_warning(f"Translation failed: {translation_status}")

            # Save results to file for single-preprocessing case
            result_file = os.path.join(img_output_dir, f"{base_filename}_result.txt")
            try:
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write(f"Image: {os.path.basename(image_path)}\n")
                    f.write(f"OCR Engine: {ocr_engine}\n")
                    f.write(f"Preprocessing: {method_name}\n")
                    f.write(f"Confidence: {confidence}\n")
                    f.write(f"Characters: {len(raw_text)}\n")
                    f.write(f"\n{'='*60}\n")
                    f.write("OCR Extracted Text:\n")
                    f.write(f"{'='*60}\n\n")
                    f.write(formatted_text)

                    if enable_ai and api_key and validation_status == "Success":
                        f.write(f"\n\n{'='*60}\n")
                        f.write("AI-Corrected Text:\n")
                        f.write(f"{'='*60}\n\n")
                        f.write(corrected_text)

                    if enable_translation and translations:
                        f.write(f"\n\n{'='*60}\n")
                        f.write("TRANSLATIONS\n")
                        f.write(f"{'='*60}\n")

                        for lang, trans_text in translations.items():
                            f.write(f"\n{lang}:\n")
                            f.write(f"{'-'*60}\n")
                            f.write(trans_text)
                            f.write("\n")
                print_success(f"\nResults saved to: {result_file}")
            except Exception as e:
                print_error(f"Failed to save results: {e}")

        else:
            print_error("No text recognized with selected preprocessing method.")


# ---------- Main Function ----------
def main():
    parser = argparse.ArgumentParser(
        description="Tamil OCR Terminal Application with AI Validation and Multi-Language Translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tamil_ocr_terminal.py image.jpg
  python tamil_ocr_terminal.py image.jpg --engine paddleocr --mode auto
  python tamil_ocr_terminal.py image.jpg --api-key YOUR_KEY --ai
  python tamil_ocr_terminal.py image.jpg --api-key YOUR_KEY --translate --languages french japanese chinese
  python tamil_ocr_terminal.py img1.jpg img2.jpg --output ./results --ai --translate
        """
    )

    parser.add_argument('images', nargs='+', help='Path(s) to input image file(s)')
    parser.add_argument('--output', '-o', default='./ocr_output',
                        help='Output directory for preprocessed images and results (default: ./ocr_output)')
    parser.add_argument('--engine', '-e', choices=['ocr_tamil', 'paddleocr'],
                        default='ocr_tamil', help='OCR engine to use (default: ocr_tamil)')
    parser.add_argument('--mode', '-m',
                        choices=['auto', 'original', 'standard', 'aggressive', 'inverted', 'otsu'],
                        default='auto',
                        help='Preprocessing mode (default: auto - tries all methods)')
    parser.add_argument('--api-key', '-k', default='',
                        help='OpenRouter API key for AI validation and translation')
    parser.add_argument('--ai', action='store_true',
                        help='Enable AI validation with Qwen2.5 72B (requires API key)')
    parser.add_argument('--translate', '-t', action='store_true',
                        help='Enable translation to other languages (requires API key)')
    parser.add_argument('--languages', '-l', nargs='+',
                        choices=['french', 'japanese', 'chinese'],
                        default=['french', 'japanese', 'chinese'],
                        help='Target languages for translation (default: all three)')
    parser.add_argument('--details', '-d', action='store_true',
                        help='Show detailed preprocessing results')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output')

    args = parser.parse_args()

    # Disable colors if requested
    if args.no_color:
        Colors.HEADER = ''
        Colors.BLUE = ''
        Colors.CYAN = ''
        Colors.GREEN = ''
        Colors.YELLOW = ''
        Colors.RED = ''
        Colors.ENDC = ''
        Colors.BOLD = ''
        Colors.UNDERLINE = ''

    # Print banner
    print_header("Tamil OCR Terminal Application")
    print(f"{Colors.CYAN}Powered by: ocr_tamil, PaddleOCR, OpenCV, and Qwen2.5 72B{Colors.ENDC}\n")

    # Validate images exist
    valid_images = []
    for img_path in args.images:
        if os.path.isfile(img_path):
            valid_images.append(img_path)
        else:
            print_error(f"Image not found: {img_path}")

    if not valid_images:
        print_error("No valid images provided. Exiting.")
        sys.exit(1)

    print_info(f"Found {len(valid_images)} valid image(s)")
    print_info(f"OCR Engine: {args.engine}")
    print_info(f"Preprocessing Mode: {args.mode}")
    print_info(f"AI Validation: {'Enabled' if args.ai else 'Disabled'}")
    print_info(f"Translation: {'Enabled' if args.translate else 'Disabled'}")

    if args.translate:
        lang_display = ', '.join([l.capitalize() for l in args.languages])
        print_info(f"Target Languages: {lang_display}")

    print_info(f"Output Directory: {args.output}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    print_success(f"Output directory ready: {args.output}")

    # Load models
    print("\n" + "="*60)
    ocr_tamil = load_ocr_model()
    paddle_ocr = load_paddle_ocr()
    print("="*60 + "\n")

    # Check API key if AI or translation is enabled
    if (args.ai or args.translate) and not args.api_key:
        print_warning("AI features enabled but no API key provided!")
        print_info("Use --api-key YOUR_KEY to enable AI validation and translation")
        args.ai = False
        args.translate = False

    # Process each image
    start_time = datetime.now()

    for idx, img_path in enumerate(valid_images, 1):
        print(f"\n{Colors.BOLD}[{idx}/{len(valid_images)}]{Colors.ENDC}")

        try:
            process_image(
                image_path=img_path,
                output_dir=args.output,
                ocr_engine=args.engine,
                preprocessing_mode=args.mode,
                api_key=args.api_key,
                enable_ai=args.ai,
                enable_translation=args.translate,
                translation_languages=args.languages,
                show_details=args.details,
                ocr_tamil=ocr_tamil,
                paddle_ocr=paddle_ocr
            )
        except Exception as e:
            print_error(f"Failed to process {img_path}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Summary
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    print_header("Processing Complete")
    print(f"{Colors.GREEN}✓ Processed {len(valid_images)} image(s) in {elapsed:.2f} seconds{Colors.ENDC}")
    print(f"{Colors.CYAN}✓ Results saved to: {os.path.abspath(args.output)}{Colors.ENDC}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Process interrupted by user{Colors.ENDC}")
        sys.exit(0)
    except Exception as e:
        print_error(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
