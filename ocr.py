import os
import base64
import json
from typing import List

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Make sure .env is loaded for this module too (idempotent, safe)
load_dotenv()


def _encode_file_to_data_url(file_obj) -> str:
    """
    Read an uploaded file (PNG/JPG) and return a data URL string
    suitable for OpenAI Vision.
    """
    file_bytes = file_obj.read()

    # reset pointer so Flask doesn't get confused later
    try:
        file_obj.stream.seek(0)
    except Exception:
        pass

    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _call_vision_on_image(data_url: str) -> List[dict]:
    """
    Call OpenAI Vision on a single screenshot and return a list of
    transaction dicts.

    Each transaction dict should look like:
      {
        "date": "2025-03-10",
        "description": "Swiggy order",
        "amount": 499.0,
        "type": "debit"
      }
    """

    # 🔑 We create the client *inside* the function, not at import time
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY missing in environment")
        return []

    client = OpenAI(api_key=api_key)

    system_prompt = (
        "You are an OCR and information-extraction assistant for Indian bank and "
        "payment SMS or notification screenshots.\n\n"
        "Your job is to read ALL transaction-like messages in the screenshot and "
        "return them as strict JSON.\n\n"
        "Rules:\n"
        "- Output a single JSON object, with a key 'transactions' that is a list.\n"
        "- Each transaction must have exactly these keys:\n"
        "    - 'date': the date of the transaction in ISO 'YYYY-MM-DD' if you can infer it, "
        "             otherwise use the string 'unknown'.\n"
        "    - 'description': short free text like 'Rent payment', 'DMart purchase', 'Swiggy order'.\n"
        "    - 'amount': a number (float) in RUPEES, with NO currency symbol and NO commas.\n"
        "    - 'type': 'debit' if money left the user, 'credit' if money came in.\n"
        "- If multiple SMS messages are shown in one screenshot, extract each as its own transaction.\n"
        "- Ignore OTP codes, order IDs and non-monetary messages.\n"
        "- If an amount is mentioned more than once, use the actual transaction amount once.\n"
    )

    user_text = (
        "Extract all the transactions you can see in this SMS screenshot.\n"
        "Return ONLY JSON, no markdown, no explanation."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": "low"},
                        },
                    ],
                },
            ],
            max_tokens=500,
        )
    except Exception as e:
        print("Vision OCR API error:", repr(e))
        return []

    try:
        content = resp.choices[0].message.content
        data = json.loads(content)
    except Exception as e:
        print("Vision JSON parse error:", repr(e))
        return []

    txs = data.get("transactions", [])
    if not isinstance(txs, list):
        return []

    cleaned: List[dict] = []
    for tx in txs:
        if not isinstance(tx, dict):
            continue

        date = str(tx.get("date", "") or "").strip()
        desc = str(tx.get("description", "") or "").strip()
        amount = tx.get("amount", 0)

        try:
            amount = float(amount)
        except Exception:
            # if the model gave weird non-numeric amount, skip this row
            continue

        tx_type = (tx.get("type") or "debit").strip().lower()
        if tx_type not in ("debit", "credit"):
            tx_type = "debit"

        if not desc and amount == 0:
            continue

        cleaned.append(
            {
                "date": date,
                "description": desc,
                "amount": amount,
                "type": tx_type,
            }
        )

    return cleaned


def process_image_files(files) -> pd.DataFrame:
    """
    Main entry point used by app.py.

    files: list of uploaded image FileStorage objects
    returns: DataFrame[date, description, amount, type]
    """
    all_rows: List[dict] = []

    for file_obj in files:
        try:
            data_url = _encode_file_to_data_url(file_obj)
            txs = _call_vision_on_image(data_url)
            if txs:
                print(
                    f"Vision OCR extracted {len(txs)} transactions "
                    f"from {getattr(file_obj, 'filename', 'image')}"
                )
                all_rows.extend(txs)
            else:
                print(
                    "Vision OCR found NO transactions for",
                    getattr(file_obj, "filename", "image"),
                )
        except Exception as e:
            print(
                "Error processing image file:",
                getattr(file_obj, "filename", "image"),
                repr(e),
            )

    if not all_rows:
        return pd.DataFrame(columns=["date", "description", "amount", "type"])

    df = pd.DataFrame(all_rows, columns=["date", "description", "amount", "type"])
    print("OCR DataFrame:\n", df)
    return df
