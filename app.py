from flask import Flask, render_template, request, redirect, url_for
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from ocr import process_image_files

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

BUDGETS_PATH = BASE_DIR / "budgets.json"


# ---------- Budgets ----------

def load_budgets():
    with open(BUDGETS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- Categorization rules ----------

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


# ---------- AI budget suggestion ----------

def suggest_budgets_ai(df_all: pd.DataFrame, budgets: dict):
    """
    Use a very simple ML model (Linear Regression) per category on
    monthly totals to project next month spending and set budget.
    Falls back to heuristic if not enough history.
    """
    df = df_all.copy()

    if "date" not in df.columns:
        # no dates at all -> use heuristics
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
            # ML: Linear Regression on time -> spend
            X = [[i] for i in range(len(cat_history))]
            y = cat_history["spent"].values
            model = LinearRegression()
            model.fit(X, y)
            next_idx = [[len(cat_history)]]
            predicted = float(model.predict(next_idx)[0])
            predicted = max(predicted, 0.0)
            suggested_budget = round(predicted * 1.1)
        else:
            # fallback: heuristic from latest spend
            latest_spent = float(cat_history["spent"].iloc[-1])
            current_budget = float(budgets.get(cat, 0))
            suggested_budget = round(max(current_budget, latest_spent * 1.1))

        suggested[cat] = suggested_budget

    # Add categories that weren't in monthly (edge case)
    for cat in df["category"].unique():
        if cat not in suggested:
            spent_total = float(df[df["category"] == cat]["amount"].sum())
            if spent_total > 0:
                suggested[cat] = round(spent_total * 1.1)

    return suggested


# ---------- Core processing ----------

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

    # parse dates if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].notna().sum() == 0:
            df["date"] = pd.NaT
    else:
        df["date"] = pd.NaT

    # Month grouping
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    if df["year_month"].isna().all():
        df["year_month"] = "Current"

    months = sorted(df["year_month"].unique())
    current_month = months[-1]
    prev_month = months[-2] if len(months) > 1 else None

    df_current = df[df["year_month"] == current_month].copy()
    df_prev = df[df["year_month"] == prev_month].copy() if prev_month else None

    budgets = load_budgets()

    # summary for current month
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

    # summary for previous month (for comparison chart)
    summary_prev = None
    if df_prev is not None and not df_prev.empty:
        summary_prev = (
            df_prev.groupby("category")["amount"]
            .sum()
            .reset_index()
            .rename(columns={"amount": "spent"})
        )

    # AI suggested budgets using full history
    suggested_ai = suggest_budgets_ai(df, budgets)

    # KPI totals
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


# ---------- Routes ----------

@app.route("/", methods=["GET", "POST"])
def index():
    # CSV upload (manual)
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
    # PNG/JPG upload -> OCR -> CSV -> dashboard
    files = request.files.getlist("images")
    if not files or files[0].filename == "":
        return redirect(url_for("index"))

    df = process_image_files(files)
    if df.empty:
        return redirect(url_for("index"))

    csv_path = DATA_DIR / "latest.csv"
    df.to_csv(csv_path, index=False)

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

    # Pie chart (current month category share)
    pie_labels = list(summary_current["category"])
    pie_values = list(summary_current["spent"])

    # Comparison bar chart: current vs previous month
    compare_labels = list(summary_current["category"])
    prev_map = {}
    if summary_prev is not None:
        prev_map = {
            row["category"]: float(row["spent"]) for _, row in summary_prev.iterrows()
        }
    compare_current = [float(row["spent"]) for _, row in summary_current.iterrows()]
    compare_prev = [prev_map.get(cat, 0.0) for cat in compare_labels]

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
    )


if __name__ == "__main__":
    app.run(debug=True)
