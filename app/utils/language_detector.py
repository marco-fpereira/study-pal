import os
from langdetect import detect
from pypdf import PdfReader
from pptx import Presentation
from docx import Document

LANG_MAP = {
    "en": "eng",
    "pt": "por",
    "es": "spa",
    "fr": "fra",
    "de": "deu",
    "it": "ita",
}

MAX_SAMPLE_CHARS = 2000  # enough for reliable detection


def _extract_text_pdf(file_path: str) -> str:
    """
    Extract text from first few pages of a PDF file
    """
    reader = PdfReader(file_path)

    sample = ""
    for page in reader.pages[:5]:
        sample += page.extract_text() or ""
        if len(sample) >= MAX_SAMPLE_CHARS:
            break

    return sample


def _extract_text_pptx(file_path: str) -> str:
    """
    Extract text from first few pages of a PowerPoint file
    """
    prs = Presentation(file_path)

    text = ""
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text += para.text + " "
        if len(text) >= MAX_SAMPLE_CHARS:
            break

    return text

def _extract_text_docx(file_path: str) -> str:
    """
    Extract text from first few pages of a Word document
    """
    doc = Document(file_path)

    text = ""
    for para in doc.paragraphs[:50]:  # first 50 paragraphs as sample
        text += para.text + " "
        if len(text) >= MAX_SAMPLE_CHARS:
            break
    
    return text


EXTRACTORS = {
    ".pdf":  _extract_text_pdf,
    ".pptx": _extract_text_pptx,
    ".ppt":  _extract_text_pptx,  # python-pptx handles .ppt too
    ".docx": _extract_text_docx,
    ".doc":  _extract_text_docx,  # python-docx handles .doc too
}

def detect_language(file_path: str) -> list[str]:
    """
    Detect dominant language from a PDF, PPTX, or DOCX file.
    Returns language list for Unstructured.
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        extractor = EXTRACTORS.get(ext)

        if extractor is None:
            return ["eng"]  # unsupported format fallback

        sample = extractor(file_path)

        if not sample.strip():
            return ["eng"]  # fallback

        detected = detect(sample)

        return [LANG_MAP.get(detected, "eng")]

    except Exception:
        return ["eng"]