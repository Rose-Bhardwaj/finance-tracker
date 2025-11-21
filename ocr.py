"""
ocr.py — OCR helpers for SMS / notification screenshots.

Used in two ways:
1) From Flask: process_image_files(files) where files is a list of
   Werkzeug FileStorage objects from an upload form.
2) As a script: python ocr.py (reads PNG/JPG from images/ folder and
   writes data/latest.csv)
"""

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytesseract
from dateutil import parser as dateparser

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_CSV = DATA_DIR / "latest.csv"

# Set this to your local Tesseract path if needed
# (comment out if Tesseract is already in PATH)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ---------- Core helpers ----------

def preprocess_sms_img(img):
    """Preprocess an SMS screenshot image (numpy array) for OCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mean_val = gray.mean()
    if mean_val < 100:  # dark background
        gray = cv2.bitwise_not(gray)

    h, w = gray.shape
    scale = 1.8 if w < 1200 else 1.3
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpen = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    _, th = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def ocr_image(img):
    """Run Tesseract OCR on a prepared image array."""
    config = "--oem 3 --psm 6 -l eng -c preserve_interword_spaces=1"
    return pytesseract.image_to_string(img, config=config)


def parse_transaction_text(text, source_name=""):
    """Parse amount/date/type/merchant from OCR text."""
    text = text.replace("\n", " ").strip()

    amt_match = re.search(r"(?:Rs\.?|INR|₹)\s*([0-9,]+(?:\.\d{1,2})?)", text, re.I)
    amount = float(amt_match.group(1).replace(",", "")) if amt_match else None

    date_match = re.search(
        r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})\b",
        text,
    )
    date = None
    if date_match:
        try:
            date = dateparser.parse(
                date_match.group(1), dayfirst=True, fuzzy=True
            ).date().isoformat()
        except Exception:
            pass

    if re.search(r"debited|spent|withdrawn|payment|paid|upi", text, re.I):
        tx_type = "debit"
    elif re.search(r"credited|received|refund|deposit", text, re.I):
        tx_type = "credit"
    else:
        tx_type = "unknown"

    merchant = ""
    m = re.search(r"\b(?:to|at|by|from|via)\s+([A-Za-z0-9@._&\- ]{2,40})", text, re.I)
    if m:
        merchant = m.group(1).strip(" .,-")

    description = merchant if merchant else text[:80]

    return {
        "filename": source_name,
        "date": date,
        "amount": amount,
        "type": tx_type,
        "merchant": merchant,
        "raw_text": text,
        "description": description,
    }


# ---------- Flask integration ----------

def process_image_files(files):
    """
    files: list of Werkzeug FileStorage objects from Flask upload.
    Returns: pandas DataFrame with parsed transactions.
    """
    rows = []

    for f in files:
        if not f or f.filename == "":
            continue

        # Read into OpenCV image
        file_bytes = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # reset pointer so future reads work if needed
        f.stream.seek(0)

        if img is None:
            continue

        pre = preprocess_sms_img(img)
        text = ocr_image(pre)
        row = parse_transaction_text(text, source_name=f.filename)
        rows.append(row)

    df = pd.DataFrame(rows)

    if "amount" in df.columns:
        df = df.dropna(subset=["amount"])

    return df


# ---------- CLI mode: process images/ folder ----------

def process_folder_to_csv(images_dir: Path):
    rows = []
    exts = {".png", ".jpg", ".jpeg"}

    if not images_dir.exists():
        print(f"❌ Folder not found: {images_dir}")
        return

    for path in images_dir.iterdir():
        if path.suffix.lower() not in exts:
            continue

        print(f"OCR: {path.name}")
        img = cv2.imread(str(path))
        if img is None:
            continue
        pre = preprocess_sms_img(img)
        text = ocr_image(pre)
        row = parse_transaction_text(text, source_name=path.name)
        rows.append(row)

    if not rows:
        print("❌ No PNG/JPG images processed.")
        return

    df = pd.DataFrame(rows)
    if "amount" in df.columns:
        df = df.dropna(subset=["amount"])

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved {len(df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    images_dir = BASE_DIR / "images"
    process_folder_to_csv(images_dir)
