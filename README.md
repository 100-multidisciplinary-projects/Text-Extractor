# Tamil OCR Application

A powerful command-line tool for extracting Tamil text from images using advanced OCR technology with AI-powered validation and correction.

## 🌟 Features

### Core Capabilities
- **Dual OCR Engines**: Choose between `ocr_tamil` and `PaddleOCR` for optimal results
- **AI-Powered Validation**: Qwen2.5 72B model corrects OCR errors and improves accuracy
- **Multiple Preprocessing Techniques**: 5 different image preprocessing methods
- **Batch Processing**: Process multiple images in a single command
- **Confidence Assessment**: Automatic quality evaluation of OCR results
- **Beautiful Terminal Output**: Color-coded, formatted results with clear visual hierarchy

---
## OCR Engines Explained
OCR Engines Explained
1. ocr_tamil (Default)
What it is: A specialized OCR library specifically designed for Tamil script recognition.
Strengths:

Optimized for Tamil Unicode characters
Better handling of complex Tamil ligatures
Lightweight and fast processing
Pre-trained on Tamil documents

Best for:

Printed Tamil text
Clear, high-contrast images
Modern Tamil fonts
Documents and books

Technical Details:

Uses deep learning models trained on Tamil datasets
Includes text detection and recognition modules
Handles diacritics and conjunct characters

2. PaddleOCR
What it is: An industrial-grade multilingual OCR system developed by Baidu, supporting 80+ languages including Tamil.
Strengths:

Robust text detection in complex layouts
Better performance on handwritten text
Advanced angle correction capabilities
Handles skewed or rotated text

Best for:

Complex document layouts
Handwritten Tamil text
Mixed Tamil-English documents
Challenging lighting conditions

Technical Details:

Uses DB (Differentiable Binarization) for text detection
CRNN (Convolutional Recurrent Neural Network) for recognition
Built-in text angle classification
Supports both GPU and CPU inference


## 📋 Requirements

### System Requirements
- Python 3.12.8
- 4GB+ RAM recommended
- CPU-based processing (no GPU required)

### Dependencies
Install all required libraries:

```bash
pip install -r requirements.txt
```

## 🚀 Installation

1. **Clone or download the repository**
   ```bash
   git clone https://github.com/100-multidisciplinary-projects/Text-Extractor
   cd Text-Extractor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python app.py image_name.jpg
   ```

---

## 💻 Usage

### Basic Usage

Process a single image:
```bash
python app.py image.jpg
```

Process multiple images:
```bash
python app.py img1.jpg img2.jpg img3.png
```

### Advanced Options

**Specify OCR Engine:**
```bash
python app.py image.jpg --engine paddleocr
```

**Choose Preprocessing Mode:**
```bash
python app.py image.jpg --mode aggressive
```

**Enable AI Validation:**
```bash
python app.py image.jpg --api-key YOUR_OPENROUTER_KEY --ai
```

**Custom Output Directory:**
```bash
python app.py image.jpg --output ./my_results
```

**Show Detailed Results:**
```bash
python app.py image.jpg --mode auto --details
```

---

## 🔧 Command-Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `images` | - | Path(s) to input image file(s) | Required |
| `--output` | `-o` | Output directory for results | `./ocr_output` |
| `--engine` | `-e` | OCR engine: `ocr_tamil` or `paddleocr` | `ocr_tamil` |
| `--mode` | `-m` | Preprocessing mode (see below) | `auto` |
| `--api-key` | `-k` | OpenRouter API key for AI validation | None |
| `--ai` | - | Enable AI validation (requires API key) | Disabled |
| `--details` | `-d` | Show detailed preprocessing results | Disabled |
| `--no-color` | - | Disable colored terminal output | Disabled |

---

## 🎨 Preprocessing Techniques

The application offers 5 preprocessing methods optimized for different image types:

### 1. **Original** (No Preprocessing)
- Uses the raw image as-is
- Best for: High-quality scanned documents with clear text
- **When to use**: Modern digital images, screenshots with crisp text

### 2. **Standard** (v1)
- Denoising + CLAHE enhancement + Adaptive thresholding
- Best for: Printed text, books, documents
- **When to use**: Standard quality scans, printed materials
- **Techniques**: 
  - Fast Non-Local Means Denoising
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Gaussian adaptive threshold

### 3. **Aggressive** (v2)
- Heavy denoising + Enhanced CLAHE + Morphological operations
- Best for: Handwritten text, faint text, low-quality images
- **When to use**: Old documents, faded text, poor lighting
- **Techniques**:
  - Aggressive denoising (h=30)
  - Mean adaptive threshold
  - Morphological closing to connect broken characters

### 4. **Inverted** (v3)
- Inverted threshold for dark backgrounds
- Best for: Dark backgrounds with light text (e.g., blackboards, dark themes)
- **When to use**: Light text on dark background, inverted images
- **Techniques**:
  - Binary inverse thresholding
  - Morphological opening to remove noise
  - Elliptical structuring element

### 5. **Otsu** (v4)
- Automatic threshold calculation using Otsu's method
- Best for: Images with clear contrast and bimodal histograms
- **When to use**: High-contrast images, clear separation between text and background
- **Techniques**:
  - Gaussian blur for noise reduction
  - Otsu's automatic thresholding
  - No morphological operations (preserves original structure)

### Auto Mode
When using `--mode auto`, the application:
1. Runs all 5 preprocessing methods
2. Saves each preprocessed image
3. Uses the "Original" method for final output
4. Displays all results if `--details` is enabled

---

## 🤖 AI Validation with Qwen2.5 72B

### What It Does
- Corrects OCR recognition errors
- Fixes spelling mistakes in Tamil text
- Ensures proper grammar and sentence structure
- Preserves the original meaning

### How to Enable

1. **Get an API Key**:
   - Sign up at [OpenRouter.ai](https://openrouter.ai/)
   - Get your API key from the dashboard

2. **Use AI Validation**:
   ```bash
   python app.py image.jpg --api-key YOUR_KEY --ai
   ```

3. **Output**:
   - Shows both OCR-extracted text and AI-corrected text
   - Saves both versions to the result file

---

## 📊 Confidence Assessment

The application automatically evaluates OCR quality:

- **Excellent** (80%+): High-quality recognition, minimal errors expected
- **Good** (65-79%): Reliable recognition, minor errors possible
- **Fair** (50-64%): Acceptable recognition, review recommended
- **Poor** (30-49%): Low quality, significant errors likely
- **Very Poor** (<30%): Very low quality, manual review required

Assessment factors:
- Text length (longer text = more reliable)
- Character quality (ratio of valid characters)
- Image clarity (Laplacian variance)

---

## 📁 Output Structure

For each processed image, the application creates:

```
ocr_output/
└── image_name/
    ├── image_name_original.png          # Original preprocessing
    ├── image_name_standard.png          # Standard preprocessing
    ├── image_name_aggressive.png        # Aggressive preprocessing
    ├── image_name_inverted.png          # Inverted preprocessing
    ├── image_name_otsu.png              # Otsu preprocessing
    └── image_name_result.txt            # Final text results
```

### Result File Contents
- Image metadata (filename, size)
- OCR engine used
- Preprocessing method applied
- Confidence score
- Character count
- OCR extracted text (formatted)
- AI-corrected text (if AI validation was enabled)

---

## 📖 Examples

### Example 1: Quick OCR
```bash
python app.py document.jpg
```
**Output**: Basic OCR with standard preprocessing, results in `./ocr_output/`

### Example 2: High-Quality Processing
```bash
python app.py old_manuscript.jpg --mode aggressive --engine paddleocr
```
**Best for**: Old documents, faded text

### Example 3: AI-Enhanced Processing
```bash
python app.py handwritten.jpg --api-key sk-or-v1-xxxxx --ai --mode auto --details
```
**Best for**: Maximum accuracy with error correction

### Example 4: Batch Processing
```bash
python app.py *.jpg --output ./batch_results --mode standard
```
**Best for**: Processing multiple images at once

### Example 5: Dark Background Text
```bash
python app.py blackboard.jpg --mode inverted
```
**Best for**: Light text on dark backgrounds

---

## 🎯 Tips for Best Results

1. **Image Quality**:
   - Use high-resolution images (300 DPI or higher for scans)
   - Ensure good lighting and minimal shadows
   - Avoid blurry or out-of-focus images

2. **Choosing OCR Engine**:
   - **ocr_tamil**: Faster, good for standard text
   - **paddleocr**: More accurate for complex layouts

3. **Preprocessing Selection**:
   - **Not sure?** Use `--mode auto` to try all methods
   - **Printed text**: Use `standard` or `original`
   - **Handwritten/faded**: Use `aggressive`
   - **Dark backgrounds**: Use `inverted`
   - **High contrast**: Use `otsu`

4. **AI Validation**:
   - Significantly improves accuracy
   - Essential for handwritten text
   - Requires internet connection
   - Uses OpenRouter API (costs apply)

---

## 📞 Support

For issues, questions, or feature requests:
- Check the troubleshooting section
- Review command-line help: `python app.py --help`
- Refer to library documentation:
  - [PaddleOCR Docs](https://github.com/PaddlePaddle/PaddleOCR)
  - [OpenCV Docs](https://docs.opencv.org/)
  - [OpenRouter API](https://openrouter.ai/docs)

---

## 🎓 Technical Details

### Image Preprocessing Pipeline
1. **Resizing**: Images > 2000px are scaled down
2. **Denoising**: Fast Non-Local Means algorithm
3. **Enhancement**: CLAHE for contrast improvement
4. **Binarization**: Adaptive or Otsu thresholding
5. **Morphology**: Opening/closing operations (method-dependent)

### OCR Processing Flow
```
Input Image → Preprocessing → OCR Engine → Text Extraction
                                              ↓
                                    Sentence Formatting
                                              ↓
                                    Confidence Assessment
                                              ↓
                                    AI Validation (optional)
                                              ↓
                                    Save Results & Display
```

---

**Happy OCR Processing! 🎉**
