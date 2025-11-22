import pytesseract
from PIL import Image
import pandas as pd
import re
import io


def extract_amount(text):
    match = re.search(r'₹\s?(\d+[.\d+]*)', text)
    if match:
        return float(match.group(1))
    # fallback: find first integer
    match = re.search(r'\b(\d{2,6})\b', text)
    if match:
        return float(match.group(1))
    return 0.0


def extract_date(text):
    match = re.search(r'(\d{2,4}[-/]\d{1,2}[-/]\d{1,2})', text)
    return match.group(1) if match else ""


def extract_type(text):
    text_low = text.lower()
    if "debited" in text_low or "spent" in text_low:
        return "debit"
    if "credited" in text_low or "received" in text_low:
        return "credit"
    return ""


def process_image_files(files):
    rows = []

    for f in files:
        img_bytes = f.read()
        img = Image.open(io.BytesIO(img_bytes))

        # OCR directly using Pillow image
        text = pytesseract.image_to_string(img)

        amount = extract_amount(text)
        date = extract_date(text)
        tx_type = extract_type(text)

        rows.append({
            "date": date,
            "description": text.strip().replace("\n", " "),
            "amount": amount,
            "type": tx_type
        })

    return pd.DataFrame(rows)
