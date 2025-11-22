import os
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
)
from openai import OpenAI

from ocr import process_image_files  # OpenAI Vision OCR

# =========================
# Setup
# =========================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "super-secret-key-change-me")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
TX_CSV_PATH = DATA_DIR / "transactions.csv"
GOALS_JSON_PATH = DATA_DIR / "goals.json"

# =========================
# Helpers
# =========================

def default_budgets():
    # Adjust if you changed categories in your UI
    return {
        "Rent": 30000,
        "Groceries": 15000,
        "Food Delivery": 5000,
        "Shopping": 5000,
        "Transport": 3000,
        "Other": 5000,
    }


def load_transactions_df() -> pd.DataFrame:
    """Load all transactions from CSV if exists."""
    if TX_CSV_PATH.exists():
        df = pd.read_csv(TX_CSV_PATH)
        # Ensure expected columns
        for col in ["date", "description", "amount", "type"]:
            if col not in df.columns:
                return pd.DataFrame(columns=["date", "description", "amount", "type"])
        # Parse dates
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        df["type"] = df["type"].fillna("debit")
        return df
    return pd.DataFrame(columns=["date", "description", "amount", "type"])


def save_transactions_df(df: pd.DataFrame):
    """Save all transactions to CSV."""
    df.to_csv(TX_CSV_PATH, index=False)


def load_goals():
    if GOALS_JSON_PATH.exists():
        try:
            with open(GOALS_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_goals(goals_list):
    with open(GOALS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(goals_list, f, ensure_ascii=False, indent=2)


def categorize(description: str) -> str:
    """Simple rule-based category mapping based on description keywords."""
    if not isinstance(description, str):
        description = str(description or "").lower()
    else:
        description = description.lower()

    if "rent" in description or "pg" in description:
        return "Rent"
    if "dmart" in description or "grocery" in description or "supermarket" in description:
        return "Groceries"
    if "swiggy" in description or "zomato" in description or "dominos" in description or "pizza" in description:
        return "Food Delivery"
    if "myntra" in description or "ajio" in description or "nykaa" in description or "shopping" in description:
        return "Shopping"
    if "uber" in description or "ola" in description or "rapido" in description or "bus" in description or "metro" in description:
        return "Transport"
    return "Other"


def build_month_comparison(transactions_df: pd.DataFrame):
    """
    Build data for 'current vs last month' bar chart.

    Returns:
      current_month_label (str),
      compare_labels (categories),
      compare_current (list[float]),
      compare_prev (list[float]),
      prev_month_exists (bool),
      demo_prev (bool: True if last month is simulated)
    """
    if transactions_df.empty:
        return "No data", [], [], [], False, False

    df = transactions_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return "No data", [], [], [], False, False

    df["month"] = df["date"].dt.to_period("M").astype(str)

    monthly_cat = (
        df.groupby(["month", "category"])["amount"]
        .sum()
        .unstack(fill_value=0)
    )

    months = sorted(monthly_cat.index)
    demo_prev = False

    if len(months) >= 2:
        # Real previous month exists
        current_month = months[-1]
        prev_month = months[-2]
        prev_month_exists = True
        compare_labels = list(monthly_cat.columns)
        compare_current = monthly_cat.loc[current_month].round(2).tolist()
        compare_prev = monthly_cat.loc[prev_month].round(2).tolist()
    else:
        # Only one real month ⇒ simulate previous month at 80%
        current_month = months[0]
        prev_month_exists = False
        demo_prev = True
        compare_labels = list(monthly_cat.columns)
        compare_current = monthly_cat.loc[current_month].round(2).tolist()
        compare_prev = [round(x * 0.8, 2) for x in compare_current]

    return current_month, compare_labels, compare_current, compare_prev, prev_month_exists, demo_prev


def build_ai_suggested_budgets(transactions_df: pd.DataFrame):
    """
    Very simple "AI-ish" suggestion:
    - Group by month & category.
    - For each category, take the last month spend and add 10% buffer.
    - If only one month, use that +10%.
    """
    if transactions_df.empty:
        return {}

    df = transactions_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return {}

    df["month"] = df["date"].dt.to_period("M").astype(str)
    monthly_cat = (
        df.groupby(["month", "category"])["amount"]
        .sum()
        .unstack(fill_value=0)
    )
    if monthly_cat.empty:
        return {}

    last_month = sorted(monthly_cat.index)[-1]
    last_row = monthly_cat.loc[last_month]

    suggested = {}
    for cat, val in last_row.items():
        # add 10% buffer
        suggested[cat] = float(round(val * 1.1, 2))

    return suggested


def extract_goals_from_text(text: str):
    """
    Very simple heuristic: any line that starts with 'Goal:' will be stored.
    """
    goals = []
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("goal:"):
            goals.append(line[5:].strip())
    return goals


# =========================
# Routes
# =========================

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Landing page:
    - Show project title & team
    - Let user upload: single screenshot or multiple screenshots
    - Configure budgets
    """
    budgets = session.get("budgets", default_budgets())
    error_msg = None

    if request.method == "POST":
        # Budgets from form (if present)
        new_budgets = {}
        for key, default_val in default_budgets().items():
            field_name = f"budget_{key.replace(' ', '_').lower()}"  # e.g. budget_rent
            try:
                val = float(request.form.get(field_name, default_val))
            except (TypeError, ValueError):
                val = default_val
            new_budgets[key] = val
        budgets = new_budgets
        session["budgets"] = budgets

        # Handle image uploads (single or folder)
        files = request.files.getlist("images")
        files = [f for f in files if f and f.filename]

        if not files:
            error_msg = "Please upload at least one screenshot."
        else:
            # OCR with OpenAI Vision
            try:
                df_new = process_image_files(files)
            except Exception as e:
                print("OCR error:", repr(e))
                df_new = pd.DataFrame(columns=["date", "description", "amount", "type"])

            if df_new.empty or df_new["amount"].fillna(0).sum() == 0:
                error_msg = (
                    "I couldn't detect any valid transaction amounts from these screenshots. "
                    "Try a clearer bank/SMS notification screenshot."
                )
            else:
                # Normalize
                df_new["date"] = pd.to_datetime(df_new["date"], errors="coerce")
                df_new["amount"] = pd.to_numeric(df_new["amount"], errors="coerce").fillna(0.0)
                df_new["type"] = df_new["type"].fillna("debit")

                # Append to stored CSV
                df_existing = load_transactions_df()
                combined = pd.concat([df_existing, df_new], ignore_index=True)
                save_transactions_df(combined)

                flash("Transactions extracted and added to your dashboard.", "success")
                return redirect(url_for("dashboard"))

    return render_template(
        "index.html",
        budgets=budgets,
        error_msg=error_msg,
    )


@app.route("/dashboard")
def dashboard():
    """
    Dashboard page:
    - KPIs
    - Pie (category share)
    - Bar (current vs last month with demo prev if needed)
    - Category summary
    - Suggested budgets
    - Goals
    - Transaction table
    - Notifications (alerts)
    """
    budgets = session.get("budgets", default_budgets())
    df_all = load_transactions_df()

    if df_all.empty:
        # No data at all
        summary = []
        pie_labels = []
        pie_values = []
        suggested_ai = {}
        alerts = []
        transactions = []
        current_month_label = "No data"
        compare_labels = []
        compare_current = []
        compare_prev = []
        prev_month_exists = False
        demo_prev = False
        total_spent = 0
        total_budget = sum(budgets.values())
        total_remaining = total_budget

    else:
        # Categorize all rows
        df_all["category"] = df_all["description"].apply(categorize)

        # Build current vs last month comparison (uses ALL data)
        (
            current_month_label,
            compare_labels,
            compare_current,
            compare_prev,
            prev_month_exists,
            demo_prev,
        ) = build_month_comparison(df_all)

        # Current period = current_month_label
        df_curr = df_all.copy()
        if current_month_label != "No data":
            df_curr["month"] = df_curr["date"].dt.to_period("M").astype(str)
            df_curr = df_curr[df_curr["month"] == current_month_label]

        # Summary per category for current period
        if df_curr.empty:
            cat_spend = {}
        else:
            cat_spend = (
                df_curr.groupby("category")["amount"]
                .sum()
                .to_dict()
            )

        summary = []
        alerts = []
        for cat, budget_val in budgets.items():
            spent = float(cat_spend.get(cat, 0.0))
            remaining = budget_val - spent
            if budget_val > 0:
                pct = (spent / budget_val) * 100
            else:
                pct = 0.0

            if spent > budget_val:
                status = "Over budget!"
                alerts.append(
                    f"You exceeded your {cat} budget: spent ₹{spent:.0f} / ₹{budget_val:.0f}."
                )
            elif spent >= 0.9 * budget_val:
                status = "Almost at limit"
                alerts.append(
                    f"You're close to your {cat} budget: spent ₹{spent:.0f} / ₹{budget_val:.0f}."
                )
            else:
                status = "On track"

            summary.append(
                {
                    "category": cat,
                    "spent": spent,
                    "budget": budget_val,
                    "remaining": remaining,
                    "pct": pct,
                    "status": status,
                }
            )

        # Total KPIs
        total_budget = float(sum(budgets.values()))
        total_spent = float(sum(cat_spend.values()))
        total_remaining = total_budget - total_spent

        # Pie chart: category share of current period
        pie_labels = list(cat_spend.keys())
        pie_values = [float(v) for v in cat_spend.values()]

        # AI suggested budgets
        suggested_ai = build_ai_suggested_budgets(df_all)

        # Transactions list for current period
        if df_curr.empty:
            transactions = []
        else:
            transactions = df_curr.sort_values("date", ascending=False).to_dict(orient="records")

    goals = load_goals()

    return render_template(
        "dashboard.html",
        # KPIs
        total_budget=total_budget,
        total_spent=total_spent,
        total_remaining=total_remaining,
        # charts
        pie_labels=pie_labels,
        pie_values=pie_values,
        compare_labels=compare_labels,
        compare_current=compare_current,
        compare_prev=compare_prev,
        # month info
        current_month=current_month_label,
        prev_month=prev_month_exists,
        demo_prev=demo_prev,
        # tables & lists
        summary=summary,
        suggested_ai=suggested_ai,
        transactions=transactions,
        alerts=alerts,
        goals=goals,
    )


@app.route("/assistant", methods=["GET"])
def assistant():
    """
    AI assistant page:
    - Shows chat history
    - Shows goals
    """
    chat_history = session.get("chat_history", [])
    goals = load_goals()
    return render_template("assistant.html", chat_history=chat_history, goals=goals)


@app.route("/ask", methods=["POST"])
def ask():
    """
    Handle AI assistant question.
    """
    user_msg = request.form.get("message", "").strip()
    if not user_msg:
        return redirect(url_for("assistant"))

    chat_history = session.get("chat_history", [])

    # Build simple context about budgets + latest month spend
    budgets = session.get("budgets", default_budgets())
    df_all = load_transactions_df()
    context_lines = []

    if not df_all.empty:
        df_all["category"] = df_all["description"].apply(categorize)
        # Use same current_month as dashboard
        (
            current_month_label,
            _cl,
            _cc,
            _cp,
            _prev_exists,
            _demo_prev,
        ) = build_month_comparison(df_all)

        df_all["month"] = df_all["date"].dt.to_period("M").astype(str)
        df_curr = df_all[df_all["month"] == current_month_label]
        cat_spend = (
            df_curr.groupby("category")["amount"].sum().to_dict()
            if not df_curr.empty
            else {}
        )
        context_lines.append(f"Current month: {current_month_label}")
        for cat, b in budgets.items():
            s = float(cat_spend.get(cat, 0.0))
            context_lines.append(f"{cat}: spent {s:.0f}, budget {b:.0f}")
    else:
        context_lines.append("No transactions uploaded yet.")

    context_str = "\n".join(context_lines)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly personal finance assistant for a student project. "
                "You see approximate category spending and budgets for the current month.\n"
                "Use this context to answer questions and suggest realistic, safe budget goals.\n"
                "If you propose concrete goals, prefix them with 'Goal:' on separate lines so they can be saved.\n\n"
                f"Context:\n{context_str}"
            ),
        }
    ]

    # Add previous chat
    for msg in chat_history:
        messages.append(
            {"role": msg["role"], "content": msg["content"]}
        )

    messages.append({"role": "user", "content": user_msg})

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=400,
        )
        reply = resp.choices[0].message.content
    except Exception as e:
        print("OpenAI error:", repr(e))
        reply = "I had an issue contacting the AI service. Please try again in a moment."

    # Update chat history
    chat_history.append({"role": "user", "content": user_msg})
    chat_history.append({"role": "assistant", "content": reply})
    session["chat_history"] = chat_history

    # Extract & store goals
    new_goals = extract_goals_from_text(reply)
    if new_goals:
        existing = load_goals()
        merged = existing + [g for g in new_goals if g not in existing]
        save_goals(merged)

    return redirect(url_for("assistant"))


# =========================
# Main
# =========================

if __name__ == "__main__":
    app.run(debug=True)
