import base64
import json
import os

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def image_file_to_data_url(file_storage):
    """Convert a Flask FileStorage image to a data URL for OpenAI Vision."""
    img_bytes = file_storage.read()
    file_storage.stream.seek(0)
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _extract_json(raw: str):
    """
    Take the raw model string (which may contain ```json fences)
    and return a parsed JSON object.
    """
    if not isinstance(raw, str):
        raise ValueError("Raw response is not a string")

    text = raw.strip()

    # Remove ```json ... ``` fences if present
    if text.startswith("```"):
        # Drop first line (``` or ```json)
        parts = text.split("\n", 1)
        if len(parts) == 2:
            text = parts[1]
        # Remove trailing ```
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0].strip()

    # Now, just to be extra safe, slice between first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    return json.loads(text)


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

If you truly cannot find a transaction amount (like ₹ or Rs or INR), set amount to 0.

CRITICAL RULES:
- Reply ONLY with raw JSON.
- Do NOT wrap the JSON in ```json``` or any other code fences.
- Do NOT add explanations or extra text.
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
    # print("RAW VISION REPLY:", raw)  # uncomment if you want to debug

    try:
        parsed = _extract_json(raw)

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
    except Exception as e:
        print("Vision JSON parse error:", repr(e), "RAW:", raw, flush=True)
        # Fall back: store raw reply in description, amount=0
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

    df = pd.DataFrame(rows)
    print("OCR DataFrame:", df, flush=True)
    return df
