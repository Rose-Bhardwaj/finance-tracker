import base64
import json
import os

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Load API key from environment
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def image_file_to_data_url(file_storage):
    """
    Convert a Flask FileStorage image to a base64 data URL for Vision.
    """
    img_bytes = file_storage.read()
    file_storage.stream.seek(0)  # reset so Flask can reuse if needed
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def call_vision_api(data_url: str) -> dict:
    """
    Call OpenAI Vision on ONE screenshot and extract ONE transaction as JSON.

    Returns a dict with keys:
    - date (string, may be empty)
    - description (short text)
    - amount (float, INR)
    - type ("debit" or "credit")
    """

    system_prompt = """
You are an OCR + bank SMS parser.

You are given a screenshot of an SMS or app notification describing a financial transaction
in Indian Rupees, e.g. from a bank, card network or payment app.

Your job:
- Read the screenshot carefully.
- Extract ONE transaction (the main one, if there are multiple).
- Return ONLY valid JSON (no markdown, no explanations) with these keys:

{
  "date": "YYYY-MM-DD or exact text like 'Yesterday' if the date is vague",
  "description": "short human-friendly description (merchant or short sentence)",
  "amount": 1234.56,
  "type": "debit" or "credit"
}

Rules:
- amount must be a number (no currency symbol) and in INR.
- If you genuinely cannot find an amount, set "amount": 0.
"""

    user_content = [
        {
            "type": "text",
            "text": "Extract the transaction from this screenshot and return ONLY JSON.",
        },
        {
            "type": "image_url",
            "image_url": {
                "url": data_url,
            },
        },
    ]

    # Force valid JSON back
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=300,
    )

    raw = response.choices[0].message.content

    # raw is guaranteed JSON string because of response_format
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


def process_image_files(files):
    """
    Takes a list of FileStorage images from Flask upload and returns a DataFrame
    with columns: date, description, amount, type.
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

    df = pd.DataFrame(rows)
    print("OCR DataFrame:", df, flush=True)
    return df
