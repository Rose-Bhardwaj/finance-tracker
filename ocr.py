import base64
import json
import os

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def image_file_to_data_url(file_storage):
    """
    Convert a Flask FileStorage image to a data URL string suitable
    for OpenAI Vision (gpt-4o-mini).
    """
    img_bytes = file_storage.read()
    file_storage.stream.seek(0)
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def call_vision_api(data_url: str) -> dict:
    """
    Calls OpenAI Vision to extract a single transaction from a screenshot.
    """
    system_prompt = """
You are an OCR and financial SMS / notification parser.

You will be shown a screenshot of an SMS or app notification that describes ONE financial transaction,
such as "Rs 500 debited from your account" or "Rs 900 credited" etc.

Your job is to read the screenshot carefully and extract:
- date: as it appears (or in YYYY-MM-DD if obvious)
- description: a short human-friendly summary (merchant or text)
- amount: numeric value only (float), in INR
- type: "debit" if money left the user, "credit" if money came to the user

Reply ONLY as valid JSON with this exact schema:

{
  "date": "...",
  "description": "...",
  "amount": 123.45,
  "type": "debit" | "credit" | ""
}
"""

    user_content = [
        {"type": "text", "text": "Read this screenshot and extract the transaction as JSON."},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=300,
    )

    raw = response.choices[0].message.content

    try:
        parsed = json.loads(raw)
        date = parsed.get("date", "") or ""
        desc = parsed.get("description", "") or ""
        amt = parsed.get("amount", 0) or 0
        typ = parsed.get("type", "") or ""

        try:
            amt = float(amt)
        except Exception:
            amt = 0.0

        return {
            "date": date,
            "description": desc,
            "amount": amt,
            "type": typ,
        }
    except Exception:
        return {
            "date": "",
            "description": str(raw),
            "amount": 0.0,
            "type": "",
        }


def process_image_files(files):
    """
    Used by app.py: takes a list of FileStorage images,
    returns DataFrame with columns: date, description, amount, type.
    """
    rows = []
    for f in files:
        try:
            data_url = image_file_to_data_url(f)
            tx = call_vision_api(data_url)
            rows.append(tx)
        except Exception as e:
            print("Vision OCR error for one image:", repr(e), flush=True)

    if not rows:
        return pd.DataFrame(columns=["date", "description", "amount", "type"])

    return pd.DataFrame(rows)
