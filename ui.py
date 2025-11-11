#!/usr/bin/env python3
"""
Tamil OCR Streamlit Application
Upload images via UI and get OCR results with AI validation and multi-language translation
"""

import os
import sys
import streamlit as st
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
import tempfile

# ---------- Torch fix ----------
import torch.serialization
if not hasattr(torch.serialization, "add_safe_globals"):
    def add_safe_globals(*args, **kwargs):
        pass
    torch.serialization.add_safe_globals = add_safe_globals


# ---------- Page Configuration ----------
st.set_page_config(
    page_title="Tamil OCR Application",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------- Custom CSS ----------
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #800020;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .success-box {
        background-color: #800020;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .warning-box {
        background-color: #800020;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
    </style>
""", unsafe_allow_html=True)


# ---------- Session State Initialization ----------
if 'ocr_tamil' not in st.session_state:
    st.session_state.ocr_tamil = None
if 'paddle_ocr' not in st.session_state:
    st.session_state.paddle_ocr = None
if 'results' not in st.session_state:
    st.session_state.results = None


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
                "HTTP-Referer": "https://tamil-ocr-streamlit.app",
                "X-Title": "Tamil OCR Streamlit Validator",
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
                "HTTP-Referer": "https://tamil-ocr-streamlit.app",
                "X-Title": "Tamil OCR Streamlit Translator",
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


# ---------- OCR Model Loaders ----------
@st.cache_resource
def load_ocr_tamil():
    """Load ocr_tamil model - cached for performance"""
    try:
        with st.spinner("Loading ocr_tamil model..."):
            ocr_tamil = OCR(detect=True)
        return ocr_tamil
    except Exception as e:
        st.error(f"Failed to load ocr_tamil: {e}")
        return None


@st.cache_resource
def load_paddle_ocr():
    """Load PaddleOCR model - cached for performance"""
    try:
        with st.spinner("Loading PaddleOCR model..."):
            paddle_ocr = PaddleOCR(lang='ta', use_angle_cls=True, use_gpu=False)
        return paddle_ocr
    except Exception as e:
        st.error(f"Failed to load PaddleOCR: {e}")
        return None


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
def run_tamil_ocr_multi(ocr_tamil, paddle_ocr, img_pil, preprocessing_methods, use_paddle):
    """Try multiple preprocessing methods and return all results"""
    all_results = {}

    for method_name, preprocess_func in preprocessing_methods.items():
        try:
            processed = preprocess_func(img_pil)

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
                'preprocessed_image': processed
            }

        except Exception as e:
            all_results[method_name] = {
                'text': f"Error: {str(e)}",
                'formatted_text': "",
                'confidence': "Failed",
                'preprocessed_image': None
            }

    return all_results


# ---------- Main Processing Function ----------
def process_image(image_pil, ocr_engine, preprocessing_mode, api_key, 
                 enable_ai, enable_translation, translation_languages,
                 ocr_tamil, paddle_ocr):
    """Process a single image and return results"""
    
    use_paddle = (ocr_engine.lower() == "paddleocr")

    # Define preprocessing methods
    methods = {
        "Original": preprocess_image_original,
        "Standard": preprocess_image_v1,
        "Aggressive": preprocess_image_v2,
        "Inverted": preprocess_image_v3,
        "Otsu": preprocess_image_v4
    }

    results = {
        'preprocessing_results': {},
        'selected_text': '',
        'formatted_text': '',
        'confidence': '',
        'corrected_text': '',
        'validation_status': 'Skipped',
        'translations': {},
        'translation_status': 'Skipped'
    }

    if preprocessing_mode.lower() == "auto":
        # Try all methods
        all_attempts = run_tamil_ocr_multi(ocr_tamil, paddle_ocr, image_pil, methods, use_paddle)
        results['preprocessing_results'] = all_attempts

        # Select best result
        original_text = all_attempts.get("Original", {}).get('text', '')
        original_confidence = all_attempts.get("Original", {}).get('confidence', '')

        if not original_text:
            # Pick best attempt
            best_text = ""
            best_conf_val = -1
            best_method_name = None
            label_map = {'Excellent': 90, 'Good': 75, 'Fair': 55, 'Poor': 35, 
                        'Very Poor': 15, 'No Text': 0, 'Failed': -1}
            
            for method, info in all_attempts.items():
                conf_label = info.get('confidence', '')
                val = label_map.get(conf_label, 50)
                if val > best_conf_val and info.get('text') and not info.get('text', '').startswith("Error"):
                    best_conf_val = val
                    best_text = info.get('text', '')
                    best_method_name = method

            if best_text:
                original_text = best_text
                original_confidence = all_attempts[best_method_name]['confidence']

        results['selected_text'] = original_text
        results['formatted_text'] = format_as_sentences(original_text)
        results['confidence'] = original_confidence

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

        processed = preprocess_func(image_pil)

        if use_paddle:
            text = run_paddle_ocr(paddle_ocr, processed)
        else:
            img_np = np.array(processed.convert("RGB"))
            text = ocr_tamil.predict(img_np)

            if isinstance(text, list):
                text = " ".join([str(x) for x in text])

        raw_text = text.strip()
        formatted_text = format_as_sentences(raw_text)
        confidence = assess_confidence(raw_text, processed)

        results['preprocessing_results'] = {
            method_name: {
                'text': raw_text,
                'formatted_text': formatted_text,
                'confidence': confidence,
                'preprocessed_image': processed
            }
        }
        results['selected_text'] = raw_text
        results['formatted_text'] = formatted_text
        results['confidence'] = confidence

    # AI Validation
    if enable_ai and api_key and results['formatted_text']:
        corrected_text, validation_status = validate_tamil_text_with_qwen(
            results['formatted_text'], api_key
        )
        results['corrected_text'] = corrected_text
        results['validation_status'] = validation_status
    else:
        results['corrected_text'] = results['formatted_text']

    # Translation
    if enable_translation and api_key and translation_languages:
        final_text = results['corrected_text'] if results['corrected_text'] else results['formatted_text']
        translations, translation_status = translate_text_with_qwen(
            final_text, translation_languages, api_key
        )
        results['translations'] = translations
        results['translation_status'] = translation_status

    return results


# ---------- Main App ----------
def main():
    # Header
    st.markdown('<p class="main-header">📝 Tamil OCR Application</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by ocr_tamil, PaddleOCR, and Qwen2.5 72B</p>', 
                unsafe_allow_html=True)

    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    
    # OCR Engine Selection
    ocr_engine = st.sidebar.selectbox(
        "OCR Engine",
        ["ocr_tamil", "paddleocr"],
        help="Select the OCR engine to use"
    )

    # Preprocessing Mode
    preprocessing_mode = st.sidebar.selectbox(
        "Preprocessing Mode",
        ["auto", "original", "standard", "aggressive", "inverted", "otsu"],
        help="Auto tries all methods and selects the best"
    )

    # API Key
    st.sidebar.subheader("🤖 AI Features")
    api_key = st.sidebar.text_input(
        "OpenRouter API Key",
        type="password",
        help="Required for AI validation and translation"
    )

    # AI Options
    enable_ai = st.sidebar.checkbox(
        "Enable AI Validation",
        value=False,
        help="Use Qwen2.5 72B to validate and correct OCR results"
    )

    enable_translation = st.sidebar.checkbox(
        "Enable Translation",
        value=False,
        help="Translate text to other languages"
    )

    translation_languages = []
    if enable_translation:
        st.sidebar.subheader("Target Languages")
        if st.sidebar.checkbox("French", value=True):
            translation_languages.append("french")
        if st.sidebar.checkbox("Japanese", value=True):
            translation_languages.append("japanese")
        if st.sidebar.checkbox("Chinese", value=True):
            translation_languages.append("chinese")

    # Load Models
    st.sidebar.divider()
    st.sidebar.subheader("🔧 Model Status")
    
    if st.session_state.ocr_tamil is None:
        st.session_state.ocr_tamil = load_ocr_tamil()
    
    if st.session_state.paddle_ocr is None:
        st.session_state.paddle_ocr = load_paddle_ocr()

    if st.session_state.ocr_tamil:
        st.sidebar.success("✓ ocr_tamil loaded")
    else:
        st.sidebar.error("✗ ocr_tamil failed to load")

    if st.session_state.paddle_ocr:
        st.sidebar.success("✓ PaddleOCR loaded")
    else:
        st.sidebar.error("✗ PaddleOCR failed to load")

    # Main Content
    st.divider()

    # File Upload
    uploaded_file = st.file_uploader(
        "Upload Tamil Text Image",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload an image containing Tamil text"
    )

    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("📸 Original Image")
            st.image(image)
            st.caption(f"Size: {image.size[0]}x{image.size[1]} pixels")

        # Process button
        if st.button("🚀 Process Image", type="primary"):
            if not st.session_state.ocr_tamil or not st.session_state.paddle_ocr:
                st.error("Models not loaded. Please check the sidebar for model status.")
                return

            if (enable_ai or enable_translation) and not api_key:
                st.warning("⚠️ AI features enabled but no API key provided!")
                enable_ai = False
                enable_translation = False

            # Process the image
            with st.spinner("Processing image... This may take a moment."):
                results = process_image(
                    image_pil=image,
                    ocr_engine=ocr_engine,
                    preprocessing_mode=preprocessing_mode,
                    api_key=api_key,
                    enable_ai=enable_ai,
                    enable_translation=enable_translation,
                    translation_languages=translation_languages,
                    ocr_tamil=st.session_state.ocr_tamil,
                    paddle_ocr=st.session_state.paddle_ocr
                )
                st.session_state.results = results

        # Display Results
        if st.session_state.results:
            results = st.session_state.results
            
            st.divider()
            st.header("📊 Results")

            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", results['confidence'])
            with col2:
                st.metric("Characters", len(results['selected_text']))
            with col3:
                validation_icon = "✓" if results['validation_status'] == "Success" else "✗"
                st.metric("AI Validation", validation_icon)

            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📝 Extracted Text", "🖼️ Preprocessed Images", 
                                               "🤖 AI Corrected", "🌍 Translations"])

            with tab1:
                st.subheader("OCR Extracted Text")
                if results['formatted_text']:
                    st.markdown(f'<div class="result-box">{results["formatted_text"]}</div>', 
                               unsafe_allow_html=True)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Text",
                        data=results['formatted_text'],
                        file_name="ocr_output.txt",
                        mime="text/plain"
                    )
                else:
                    st.warning("No text was extracted from the image.")

            with tab2:
                st.subheader("Preprocessing Results")
                if results['preprocessing_results']:
                    for method, result in results['preprocessing_results'].items():
                        with st.expander(f"{method} - Confidence: {result['confidence']}", 
                                       expanded=(method == "Original")):
                            if result['preprocessed_image']:
                                col1, col2 = st.columns([1, 1])
                                with col1:
                                    st.image(result['preprocessed_image'], 
                                           caption=f"{method} Preprocessing",
                                           )
                                with col2:
                                    st.write("**Extracted Text Preview:**")
                                    preview = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
                                    st.text_area("", preview, height=200, key=f"preview_{method}")
                            else:
                                st.error(f"Processing failed: {result['text']}")

            with tab3:
                st.subheader("AI-Corrected Text")
                if enable_ai and results['validation_status'] == "Success":
                    st.markdown(f'<div class="success-box">{results["corrected_text"]}</div>', 
                               unsafe_allow_html=True)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Corrected Text",
                        data=results['corrected_text'],
                        file_name="corrected_output.txt",
                        mime="text/plain"
                    )
                    
                    # Show comparison
                    if results['formatted_text'] != results['corrected_text']:
                        st.info("💡 AI made corrections to improve accuracy")
                        with st.expander("View Original vs Corrected"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Original OCR:**")
                                st.text_area("", results['formatted_text'], height=200, key="orig_comparison")
                            with col2:
                                st.write("**AI Corrected:**")
                                st.text_area("", results['corrected_text'], height=200, key="corrected_comparison")
                    else:
                        st.success("✓ No corrections needed - OCR was accurate!")
                        
                elif enable_ai and results['validation_status'] != "Success":
                    st.error(f"AI validation failed: {results['validation_status']}")
                    st.info("Showing OCR extracted text instead:")
                    st.markdown(f'<div class="warning-box">{results["formatted_text"]}</div>', 
                               unsafe_allow_html=True)
                else:
                    st.info("AI validation was not enabled. Enable it in the sidebar to use this feature.")
                    st.markdown(f'<div class="result-box">{results["formatted_text"]}</div>', 
                               unsafe_allow_html=True)

            with tab4:
                st.subheader("Translations")
                if enable_translation and results['translations']:
                    if results['translation_status'] == "Success":
                        st.success(f"✓ Successfully translated to {len(results['translations'])} language(s)")
                        
                        # Display translations
                        lang_icons = {
                            'FRENCH': '🇫🇷',
                            'JAPANESE': '🇯🇵',
                            'CHINESE': '🇨🇳'
                        }
                        
                        for lang, trans_text in results['translations'].items():
                            icon = lang_icons.get(lang, '🌐')
                            st.markdown(f"### {icon} {lang.capitalize()} Translation")
                            st.markdown(f'<div class="result-box">{trans_text}</div>', 
                                       unsafe_allow_html=True)
                            
                            # Download button for each translation
                            st.download_button(
                                label=f"📥 Download {lang.capitalize()} Translation",
                                data=trans_text,
                                file_name=f"translation_{lang.lower()}.txt",
                                mime="text/plain",
                                key=f"download_{lang}"
                            )
                            st.divider()
                    else:
                        st.error(f"Translation failed: {results['translation_status']}")
                        if results['translations']:
                            st.warning("Partial translations available:")
                            for lang, trans_text in results['translations'].items():
                                with st.expander(f"{lang} Translation"):
                                    st.write(trans_text)
                else:
                    st.info("Translation was not enabled. Enable it in the sidebar and select target languages to use this feature.")

            # Export all results
            st.divider()
            st.subheader("📦 Export Complete Results")
            
            export_data = f"""Tamil OCR Results
{'='*60}

Image Information:
- Size: {image.size[0]}x{image.size[1]} pixels
- OCR Engine: {ocr_engine}
- Preprocessing Mode: {preprocessing_mode}
- Confidence: {results['confidence']}
- Characters: {len(results['selected_text'])}

{'='*60}
OCR EXTRACTED TEXT:
{'='*60}

{results['formatted_text']}

"""
            
            if enable_ai and results['validation_status'] == "Success":
                export_data += f"""
{'='*60}
AI-CORRECTED TEXT:
{'='*60}

{results['corrected_text']}

"""
            
            if enable_translation and results['translations']:
                export_data += f"""
{'='*60}
TRANSLATIONS:
{'='*60}

"""
                for lang, trans_text in results['translations'].items():
                    export_data += f"""
{lang}:
{'-'*60}
{trans_text}

"""
            
            st.download_button(
                label="📥 Download Complete Report",
                data=export_data,
                file_name=f"tamil_ocr_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                
            )

    else:
        # Show instructions when no image is uploaded
        st.info("👆 Upload an image containing Tamil text to get started")
        
        st.markdown("""
        ### How to use:
        
        1. **Upload an Image**: Click the file uploader and select an image with Tamil text
        2. **Configure Settings**: Use the sidebar to:
           - Choose OCR engine (ocr_tamil or PaddleOCR)
           - Select preprocessing mode (auto recommended)
           - Add your OpenRouter API key for AI features
           - Enable AI validation and/or translation
        3. **Process**: Click the "Process Image" button
        4. **View Results**: Explore different tabs for:
           - Extracted text
           - Preprocessed images
           - AI corrections
           - Translations
        5. **Export**: Download individual results or complete report
        
        ### Features:
        
        - ✨ **Multiple Preprocessing**: Auto mode tries 5 different preprocessing methods
        - 🤖 **AI Validation**: Uses Qwen2.5 72B to correct OCR errors
        - 🌍 **Multi-Language Translation**: Translate to French, Japanese, and Chinese
        - 📊 **Confidence Scoring**: Get quality metrics for OCR results
        - 💾 **Export Options**: Download text, corrections, translations, or full report
        
        ### Tips:
        
        - Use **auto** preprocessing mode for best results
        - Clear, high-contrast images work best
        - Enable AI validation for improved accuracy
        - Add API key from [OpenRouter](https://openrouter.ai) for AI features
        """)

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Tamil OCR Application | Powered by ocr_tamil, PaddleOCR, OpenCV, and Qwen2.5 72B</p>
        <p style='font-size: 0.9em;'>Built with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()