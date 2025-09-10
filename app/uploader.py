from io import BytesIO
from typing import List, Tuple, Optional
from pypdf import PdfReader
import chardet
import csv
import requests
import os
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes


try:
    import docx  # python-docx
except Exception:
    docx = None

# ---- 基本文字切塊 ----
def chunk_text(txt: str, chunk_size: int = 1200, overlap: int = 120) -> List[str]:
    txt = (txt or "").strip()
    if not txt:
        return []
    chunks = []
    i = 0
    n = len(txt)
    while i < n:
        end = min(i + chunk_size, n)
        chunks.append(txt[i:end])
        if end >= n:
            break
        i = end - overlap
        if i < 0:
            i = 0
    return chunks

# ---- 檔案類型判斷 ----
def ext_of(filename: str) -> str:
    _, ext = os.path.splitext(filename.lower())
    return ext.lstrip(".")

# ---- 各格式解析成文字 ----
def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    texts = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        texts.append(f"[Page {i+1}]\n{t}")
    return "\n\n".join(texts)

def extract_text_from_docx(file_bytes: bytes) -> str:
    if not docx:
        return ""
    bio = BytesIO(file_bytes)
    d = docx.Document(bio)
    return "\n".join(p.text for p in d.paragraphs)

def extract_text_from_txt(file_bytes: bytes) -> str:
    enc = chardet.detect(file_bytes).get("encoding") or "utf-8"
    try:
        return file_bytes.decode(enc, errors="ignore")
    except Exception:
        return file_bytes.decode("utf-8", errors="ignore")

def extract_text_from_csv(file_bytes: bytes) -> str:
    enc = chardet.detect(file_bytes).get("encoding") or "utf-8"
    f = BytesIO(file_bytes)
    lines = f.read().decode(enc, errors="ignore").splitlines()
    reader = csv.reader(lines)
    out = []
    for row in reader:
        out.append(", ".join(row))
    return "\n".join(out)

# ---- URL 抓取（支援 pdf/txt/csv/docx）----
def fetch_bytes_from_url(url: str) -> Tuple[Optional[bytes], Optional[str]]:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    content_type = r.headers.get("content-type", "")
    return r.content, content_type

def extract_text_by_ext(file_bytes: bytes, filename_or_ct: str) -> str:
    # filename 或 content-type 來推斷
    hint = filename_or_ct.lower()
    if ".pdf" in hint or "application/pdf" in hint:
        return extract_text_from_pdf(file_bytes)
    if ".docx" in hint or "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in hint:
        return extract_text_from_docx(file_bytes)
    if ".csv" in hint or "text/csv" in hint:
        return extract_text_from_csv(file_bytes)
    # 預設當成純文字
    return extract_text_from_txt(file_bytes)
def ocr_image_bytes(file_bytes: bytes, lang: str = "eng") -> str:
    img = Image.open(BytesIO(file_bytes))
    return pytesseract.image_to_string(img, lang=lang)

def ocr_pdf_bytes(file_bytes: bytes, lang: str = "eng", poppler_path: str | None = None) -> str:
    # 將 PDF 每頁轉圖後 OCR
    pages = convert_from_bytes(file_bytes, dpi=300, poppler_path=poppler_path)
    texts = []
    for i, img in enumerate(pages, start=1):
        t = pytesseract.image_to_string(img, lang=lang)
        texts.append(f"[Page {i}]\n{t}")
    return "\n\n".join(texts)
