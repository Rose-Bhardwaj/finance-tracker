import os
import json
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
)
import pandas as pd
from sklearn.linear_model import LinearRegression

from ocr import process_image_files

from openai import OpenAI
from dotenv import load_dotenv

# ========= Setup =========
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

BUDGETS_PATH = BASE_DIR / "budgets.json"
GOALS_PATH = BASE_DIR / "goals.json"

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")


# ========= Helpers: Budgets & Goals =========
def load_budgets():
    with open(BUDGETS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_goals():
    if not GOALS_PATH.exists():
        return []
    with open(GOALS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_goals(goals):
    with open(GOALS_PATH, "w", encoding="utf-8") as f:
        json.dump(goals, f, indent=2)


# ========= Categorization =========
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


# ========= Budget suggestion (ML) =========
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


# ========= Insights & notifications =========
def generate_insights(result):
    summary_current = result["summary_current"]
    summary_prev = result["summary_prev"]

    insights = []

    overspent = summary_current[summary_current["spent"] > summary_current["budget"]]
    if not overspent.empty:
        cats = ", ".join(overspent["category"])
        insights.append(
            f"You are over budget in: {cats}. Try reducing discretionary spend or adjusting budgets."
        )
    else:
        insights.append(
            "You are within budget in all configured categories right now. Nice control over your spending!"
        )

    almost = summary_current[
        (summary_current["budget"] > 0)
        & (summary_current["spent"] / summary_current["budget"] >= 0.8)
        & (summary_current["spent"] <= summary_current["budget"])
    ]
    if not almost.empty:
        names = ", ".join(almost["category"])
        insights.append(
            f"You're close to your budget limit for: {names}. Slow down spending there for the rest of the month."
        )

    if not summary_current.empty:
        top_row = summary_current.sort_values("spent", ascending=False).iloc[0]
        insights.append(
            f"Your biggest spending category this period is {top_row['category']} "
            f"at roughly ₹{int(top_row['spent'])}."
        )

    summary_prev = result["summary_prev"]
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

    under = summary_current[
        (summary_current["budget"] > 0)
        & (summary_current["spent"] / summary_current["budget"] < 0.4)
    ]
    if not under.empty:
        cats = ", ".join(under["category"].head(3))
        insights.append(
            f"You are using less than 40% of the budget in: {cats}. "
            f"Consider reallocating some of that to savings or other goals."
        )

    return insights


def generate_notifications(summary_current):
    """Return list of warning strings for near/over budget."""
    notes = []
    for _, row in summary_current.iterrows():
        if row["budget"] <= 0:
            continue
        ratio = row["spent"] / row["budget"]
        if ratio >= 1.0:
            notes.append(
                f"⚠ You exceeded your {row['category']} budget "
                f"({int(row['spent'])} / {int(row['budget'])})."
            )
        elif ratio >= 0.9:
            notes.append(
                f"⚠ You are nearing your {row['category']} budget "
                f"({int(ratio * 100)}% used)."
            )
    return notes


# ========= Core CSV processing =========
def process_transactions(csv_path: Path):
    df = pd.read_csv(csv_path)

    if "description" not in df.columns:
        if "merchant" in df.columns:
            df["description"] = df["merchant"].fillna("").astype(str)
        elif "raw_text" in df.columns:
            df["description"] = df["raw_text"].fillna("").astype(str)
        else:
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
    }


# ========= Routes: 3 pages =========

# --- Page 1: Welcome + Upload ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST" and "file" in request.files:
        file = request.files.get("file")
        if not file or file.filename == "":
            return redirect(url_for("index"))
        save_path = DATA_DIR / "latest.csv"
        file.save(save_path)
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/upload-images", methods=["POST"])
def upload_images():
    files = request.files.getlist("images")
    if not files or files[0].filename == "":
        return redirect(url_for("index"))

    df = process_image_files(files)
    if df.empty:
        return redirect(url_for("index"))

    csv_path = DATA_DIR / "latest.csv"
    df.to_csv(csv_path, index=False)

    return redirect(url_for("dashboard"))


# --- Page 2: Dashboard ---
@app.route("/dashboard")
def dashboard():
    csv_path = DATA_DIR / "latest.csv"
    if not csv_path.exists():
        return redirect(url_for("index"))

    result = process_transactions(csv_path)

    summary_current = result["summary_current"]
    summary_prev = result["summary_prev"]
    budgets = result["budgets"]

    pie_labels = list(summary_current["category"])
    pie_values = list(summary_current["spent"])

    compare_labels = list(summary_current["category"])
    prev_map = {}
    if summary_prev is not None:
        prev_map = {
            row["category"]: float(row["spent"]) for _, row in summary_prev.iterrows()
        }
    compare_current = [float(row["spent"]) for _, row in summary_current.iterrows()]
    compare_prev = [prev_map.get(cat, 0.0) for cat in compare_labels]

    insights = generate_insights(result)
    notifications = generate_notifications(summary_current)

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
        notifications=notifications,
    )


# --- Page 3: AI Assistant ---
@app.route("/assistant")
def assistant():
    csv_path = DATA_DIR / "latest.csv"
    if not csv_path.exists():
        return redirect(url_for("index"))

    result = process_transactions(csv_path)
    insights = generate_insights(result)
    goals = load_goals()

    chat_history = session.get("chat_history", [])

    return render_template(
        "assistant.html",
        chat_history=chat_history,
        goals=goals,
        insights=insights,
    )


@app.route("/assistant/ask", methods=["POST"])
def assistant_ask():
    message = request.form.get("message", "").strip()
    if not message:
        return redirect(url_for("assistant"))

    csv_path = DATA_DIR / "latest.csv"
    if not csv_path.exists():
        return redirect(url_for("index"))

    result = process_transactions(csv_path)
    insights = generate_insights(result)
    summary_current = result["summary_current"].to_dict(orient="records")
    suggested_ai = result["suggested_ai"]

    system_prompt = f"""
You are an AI personal finance agent inside a web app.
You see the user's spending, budgets, insights, and suggested next-month budgets.

Data:

1) Current month summary (category, spent, budget, remaining, status):
{json.dumps(summary_current, indent=2)}

2) Insights the analytics engine already found:
{json.dumps(insights, indent=2)}

3) Suggested budgets for next month:
{json.dumps(suggested_ai, indent=2)}

You must reply ONLY in valid JSON with two keys:
- "reply": a clear, friendly natural-language answer for the user.
- "new_goals": an array of short goal strings to add to the user's goals list (can be empty).

Examples of good goals:
- "Keep dining-out spending under ₹3000 next month"
- "Save ₹5000 into emergency fund"
- "Reduce movie spending by 50%"
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            max_tokens=350,
        )
        raw_content = response.choices[0].message.content
        parsed = json.loads(raw_content)
        reply = parsed.get("reply", "Sorry, I had trouble generating a reply.")
        new_goals = parsed.get("new_goals", [])
    except Exception as e:
        reply = (
            "I had an issue understanding that or contacting the AI service. "
            "Please try again in a moment."
        )
        new_goals = []

    chat_history = session.get("chat_history", [])
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": reply})
    session["chat_history"] = chat_history

    if new_goals:
        goals = load_goals()
        for g in new_goals:
            goals.append({"text": g, "status": "active"})
        save_goals(goals)

    return redirect(url_for("assistant"))


@app.route("/assistant/goals/add", methods=["POST"])
def add_goal():
    text = request.form.get("goal_text", "").strip()
    if not text:
        return redirect(url_for("assistant"))
    goals = load_goals()
    goals.append({"text": text, "status": "active"})
    save_goals(goals)
    return redirect(url_for("assistant"))


@app.route("/assistant/goals/toggle", methods=["POST"])
def toggle_goal():
    index = int(request.form.get("index", -1))
    goals = load_goals()
    if 0 <= index < len(goals):
        goals[index]["status"] = (
            "done" if goals[index]["status"] == "active" else "active"
        )
        save_goals(goals)
    return redirect(url_for("assistant"))


@app.route("/assistant/goals/clear", methods=["POST"])
def clear_goals():
    save_goals([])
    return redirect(url_for("assistant"))


if __name__ == "__main__":
    app.run(debug=True)
