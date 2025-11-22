from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
)
import json
from pathlib import Path
import os

import pandas as pd
from sklearn.linear_model import LinearRegression

from ocr import process_image_files  # OpenAI Vision OCR

from openai import OpenAI
from dotenv import load_dotenv

# ------------ Setup ------------

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
app.secret_key = "super-secret-key-change-this"  # needed for session chat history

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

BUDGETS_PATH = BASE_DIR / "budgets.json"
GOALS_PATH = BASE_DIR / "goals.json"


# ------------ Helpers ------------

def load_budgets():
    with open(BUDGETS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_goals():
    if not GOALS_PATH.exists():
        return []
    with open(GOALS_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return []


def save_goals(goals):
    with open(GOALS_PATH, "w", encoding="utf-8") as f:
        json.dump(goals, f, indent=2)


def categorize_transaction(description: str):
    desc = str(description).lower()

    if "rent" in desc or "landlord" in desc:
        return "Rent"

    if any(k in desc for k in ["grocery", "groceries", "dmart", "d-mart", "more", "big bazaar", "supermarket"]):
        return "Groceries"

    if any(k in desc for k in ["inox", "pvr", "movie", "cinema", "bookmyshow"]):
        return "Movies"

    if any(k in desc for k in ["swiggy", "zomato", "restaurant", "cafe", "pizza", "burger"]):
        return "Dining Out"

    if any(k in desc for k in ["amazon", "flipkart", "myntra", "ajio", "zara", "hm"]):
        return "Shopping"

    return "Miscellaneous"


def suggest_budgets_ai(df_all: pd.DataFrame, budgets: dict):
    df = df_all.copy()

    if "date" not in df.columns:
        return {}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].notna().sum() == 0:
        return {}

    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    monthly = (
        df.groupby(["year_month", "category"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "spent"})
    )

    suggested = {}

    for cat in monthly["category"].unique():
        cat_history = monthly[monthly["category"] == cat].sort_values("year_month")
        if len(cat_history) >= 3:
            X = [[i] for i in range(len(cat_history))]
            y = cat_history["spent"].values
            model = LinearRegression()
            model.fit(X, y)
            next_idx = [[len(cat_history)]]
            predicted = float(model.predict(next_idx)[0])
            predicted = max(predicted, 0.0)
            suggested_budget = round(predicted * 1.1)
        else:
            latest_spent = float(cat_history["spent"].iloc[-1])
            current_budget = float(budgets.get(cat, 0))
            suggested_budget = round(max(current_budget, latest_spent * 1.1))

        suggested[cat] = suggested_budget

    for cat in df["category"].unique():
        if cat not in suggested:
            spent_total = float(df[df["category"] == cat]["amount"].sum())
            if spent_total > 0:
                suggested[cat] = round(spent_total * 1.1)

    return suggested


def generate_insights(result):
    summary_current = result["summary_current"]
    summary_prev = result["summary_prev"]

    insights = []

    overspent = summary_current[summary_current["spent"] > summary_current["budget"]]
    if not overspent.empty:
        cats = ", ".join(overspent["category"])
        insights.append(
            f"You are currently over budget in: {cats}. "
            f"Try reducing discretionary spend here or increasing the budget if these are essentials."
        )
    else:
        insights.append(
            "You are within budget in all configured categories right now. "
            "Nice control over your spending!"
        )

    almost = summary_current[
        (summary_current["budget"] > 0)
        & (summary_current["spent"] / summary_current["budget"] >= 0.9)
        & (summary_current["spent"] <= summary_current["budget"])
    ]
    if not almost.empty:
        names = ", ".join(almost["category"])
        insights.append(
            f"You're very close to your budget limit for: {names}. "
            f"Be extra careful with these for the rest of the month."
        )

    if not summary_current.empty:
        top_row = summary_current.sort_values("spent", ascending=False).iloc[0]
        insights.append(
            f"Your biggest spending category this period is {top_row['category']} "
            f"at roughly ₹{int(top_row['spent'])}."
        )

    if summary_prev is not None and not summary_prev.empty:
        merged = summary_current.merge(
            summary_prev,
            on="category",
            how="left",
            suffixes=("_current", "_prev"),
        )
        merged["spent_prev"] = merged["spent_prev"].fillna(0)
        merged["delta"] = merged["spent_current"] - merged["spent_prev"]
        most_increased = merged.sort_values("delta", ascending=False).iloc[0]
        if most_increased["delta"] > 0:
            insights.append(
                f"Compared to {result['prev_month']}, your spending in {most_increased['category']} "
                f"increased by about ₹{int(most_increased['delta'])}."
            )

    return insights


def process_transactions(csv_path: Path):
    df = pd.read_csv(csv_path)

    if "description" not in df.columns:
        df["description"] = ""
    df["description"] = df["description"].fillna("").astype(str)

    if "amount" not in df.columns:
        raise ValueError("CSV must have an 'amount' column.")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    df["category"] = df["description"].apply(categorize_transaction)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].notna().sum() == 0:
            df["date"] = pd.NaT
    else:
        df["date"] = pd.NaT

    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    if df["year_month"].isna().all():
        df["year_month"] = "Current"

    months = sorted(df["year_month"].unique())
    current_month = months[-1]
    prev_month = months[-2] if len(months) > 1 else None

    df_current = df[df["year_month"] == current_month].copy()
    df_prev = df[df["year_month"] == prev_month].copy() if prev_month else None

    budgets = load_budgets()

    summary_current = (
        df_current.groupby("category")["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "spent"})
    )
    summary_current["budget"] = summary_current["category"].map(budgets).fillna(0)
    summary_current["remaining"] = summary_current["budget"] - summary_current["spent"]

    def status_row(row):
        if row["budget"] == 0:
            return "No budget set"
        ratio = row["spent"] / row["budget"] if row["budget"] > 0 else 0
        if ratio >= 1.0:
            return "Over budget!"
        elif ratio >= 0.9:
            return "Almost at limit"
        else:
            return "OK"

    summary_current["status"] = summary_current.apply(status_row, axis=1)

    summary_prev = None
    if df_prev is not None and not df_prev.empty:
        summary_prev = (
            df_prev.groupby("category")["amount"]
            .sum()
            .reset_index()
            .rename(columns={"amount": "spent"})
        )

    suggested_ai = suggest_budgets_ai(df, budgets)

    total_spent = float(summary_current["spent"].sum())
    total_budget = sum(budgets.get(cat, 0) for cat in budgets)
    total_remaining = total_budget - total_spent

    alerts = []
    for _, row in summary_current.iterrows():
        if row["status"] == "Over budget!":
            alerts.append(
                f"You have exceeded your budget in {row['category']} "
                f"(spent ₹{int(row['spent'])} vs budget ₹{int(row['budget'])})."
            )
        elif row["status"] == "Almost at limit":
            alerts.append(
                f"You are very close to your budget limit in {row['category']} "
                f"(spent ₹{int(row['spent'])} of ₹{int(row['budget'])})."
            )

    return {
        "df_all": df,
        "df_current": df_current,
        "summary_current": summary_current,
        "summary_prev": summary_prev,
        "budgets": budgets,
        "suggested_ai": suggested_ai,
        "current_month": current_month,
        "prev_month": prev_month,
        "total_spent": total_spent,
        "total_budget": total_budget,
        "total_remaining": total_remaining,
        "alerts": alerts,
    }


# ------------ Routes ------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload-images", methods=["POST"])
def upload_images():
    files = request.files.getlist("images")
    if not files or files[0].filename == "":
        return redirect(url_for("index"))

    try:
        df = process_image_files(files)
    except Exception as e:
        print("OCR error:", e, flush=True)
        return render_template(
            "index.html",
            ocr_error=(
                "There was an issue running OCR on your screenshots. "
                "Please try again or check your API key / internet connection."
            ),
        )

    if df.empty or df["amount"].sum() == 0:
        return render_template(
            "index.html",
            ocr_error=(
                "I couldn't detect any transaction amounts from those screenshots. "
                "Try a clearer SMS screenshot where the amount is visible."
            ),
        )

    csv_path = DATA_DIR / "latest.csv"
    df.to_csv(csv_path, index=False)

    session["chat_history"] = []

    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    csv_path = DATA_DIR / "latest.csv"
    if not csv_path.exists():
        return redirect(url_for("index"))

    result = process_transactions(csv_path)

    summary_current = result["summary_current"]
    summary_prev = result["summary_prev"]
    budgets = result["budgets"]
    goals = load_goals()

    pie_labels = list(summary_current["category"])
    pie_values = list(summary_current["spent"])

    compare_labels = list(summary_current["category"])
    prev_map = {}
    if summary_prev is not None:
        prev_map = {row["category"]: float(row["spent"]) for _, row in summary_prev.iterrows()}
    compare_current = [float(row["spent"]) for _, row in summary_current.iterrows()]
    compare_prev = [prev_map.get(cat, 0.0) for cat in compare_labels]

    insights = generate_insights(result)

    return render_template(
        "dashboard.html",
        transactions=result["df_current"].to_dict(orient="records"),
        summary=summary_current.to_dict(orient="records"),
        budgets=budgets,
        suggested_ai=result["suggested_ai"],
        current_month=result["current_month"],
        prev_month=result["prev_month"],
        total_spent=result["total_spent"],
        total_budget=result["total_budget"],
        total_remaining=result["total_remaining"],
        pie_labels=pie_labels,
        pie_values=pie_values,
        compare_labels=compare_labels,
        compare_current=compare_current,
        compare_prev=compare_prev,
        insights=insights,
        goals=goals,
        alerts=result["alerts"],
    )


@app.route("/assistant")
def assistant():
    csv_path = DATA_DIR / "latest.csv"
    has_data = csv_path.exists()
    chat_history = session.get("chat_history", [])
    goals = load_goals()

    quick_stats = None
    insights = []
    if has_data:
        result = process_transactions(csv_path)
        quick_stats = {
            "total_spent": result["total_spent"],
            "total_budget": result["total_budget"],
            "total_remaining": result["total_remaining"],
            "current_month": result["current_month"],
        }
        insights = generate_insights(result)[:3]

    return render_template(
        "assistant.html",
        chat_history=chat_history,
        has_data=has_data,
        goals=goals,
        quick_stats=quick_stats,
        insights=insights,
    )


@app.route("/ask", methods=["POST"])
def ask():
    user_message = request.form.get("message", "").strip()
    if not user_message:
        return redirect(url_for("assistant"))

    csv_path = DATA_DIR / "latest.csv"
    if not csv_path.exists():
        result = None
        summary_current = []
        insights = []
        suggested_ai = {}
    else:
        result = process_transactions(csv_path)
        summary_current = result["summary_current"].to_dict(orient="records")
        insights = generate_insights(result)
        suggested_ai = result["suggested_ai"]

    goals_existing = load_goals()

    system_prompt = f"""
You are an AI personal finance assistant inside a web app.
You have access to the user's current spending summary, budgets, and AI-predicted budgets.

Data you have (may be empty if user hasn't uploaded yet):

1) Current month category summary (list of dicts with keys: category, spent, budget, remaining, status):
{json.dumps(summary_current, indent=2)}

2) High-level insights your analysis engine already generated:
{json.dumps(insights, indent=2)}

3) Predicted budgets for next month (per category):
{json.dumps(suggested_ai, indent=2)}

4) Existing user goals:
{json.dumps(goals_existing, indent=2)}

You must reply ONLY in valid JSON with this exact structure:

{{
  "answer": "<natural language answer to the user's question>",
  "goals": ["<goal 1>", "<goal 2>", "..."]
}}

- "goals" should be a list of clear, short, actionable goals.
- If you don't want to add or change any goals, return "goals": [].
- Do not include any other keys.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=350,
        )
        raw_reply = response.choices[0].message.content
    except Exception as e:
        print("OpenAI error:", repr(e), flush=True)
        raw_reply = json.dumps({
            "answer": (
                "I had an issue contacting the AI service just now. "
                "Please check that the API key is valid and try again in a moment."
            ),
            "goals": []
        })

    try:
        parsed = json.loads(raw_reply)
        answer = parsed.get("answer", raw_reply)
        new_goals = parsed.get("goals", [])
    except Exception:
        answer = raw_reply
        new_goals = []

    if new_goals:
        combined = goals_existing[:]
        for g in new_goals:
            if g and g not in combined:
                combined.append(g)
        save_goals(combined)

    history = session.get("chat_history", [])
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": answer})
    history = history[-20:]
    session["chat_history"] = history

    return redirect(url_for("assistant"))


@app.route("/clear-chat")
def clear_chat():
    session["chat_history"] = []
    return redirect(url_for("assistant"))


if __name__ == "__main__":
    app.run(debug=True)
