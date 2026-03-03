# pdf_utils.py
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from:
    1. Normal text PDFs
    2. Image-based PDFs using OCR
    """
    text = ""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

    for page in doc:
        # 1️⃣ Try normal text extraction
        page_text = page.get_text()
        if page_text.strip():
            text += page_text + "\n"
        else:
            # 2️⃣ OCR fallback (scanned pages)
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(img)
            text += ocr_text + "\n"

    return text.strip()
