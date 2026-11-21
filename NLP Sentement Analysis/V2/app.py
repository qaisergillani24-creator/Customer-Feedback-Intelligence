import pandas as pd
from flask import Flask, Response, redirect, render_template, request, url_for

from ai_client import AIError, analyze_batch, analyze_review
from analytics import add_results, compute_kpis, product_summary

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB
MAX_ROWS = 200
REQUIRED = ["customer_id", "product", "feedback"]

# Aakhri analysis memory mein (download ke liye)
LAST = {"df": None}


def load_csv(file):
    """Upload hui CSV ko check kar ke saaf DataFrame wapas karta hai."""
    if not file or not file.filename:
        raise ValueError("Pehle aik CSV file chunein.")
    if not file.filename.lower().endswith(".csv"):
        raise ValueError("Sirf .csv file qabil-e-qubool hai.")
    try:
        df = pd.read_csv(file.stream, encoding="utf-8-sig", dtype=str)
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file khali hai.")
    except (pd.errors.ParserError, UnicodeDecodeError):
        raise ValueError("CSV file parh nahi saka. Format check karein (UTF-8 hona chahiye).")

    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            "Ye columns missing hain: " + ", ".join(missing)
            + ". Zaroori columns: customer_id, product, feedback."
        )

    df = df[REQUIRED].copy()
    df["customer_id"] = df["customer_id"].fillna("").str.strip()
    df["product"] = df["product"].fillna("").str.strip().replace("", "Unknown")
    df["feedback"] = df["feedback"].fillna("").str.strip()
    df = df[df["feedback"] != ""].reset_index(drop=True)

    if df.empty:
        raise ValueError("Is file mein koi feedback nahi mili.")
    if len(df) > MAX_ROWS:
        raise ValueError(
            f"File mein {len(df)} reviews hain. Free tier ke liye max {MAX_ROWS} reviews allowed hain."
        )
    return df


def build_chart_data(kpis, summary_df):
    return {
        "sentiment": [kpis["positive"], kpis["neutral"], kpis["negative"]],
        "products": summary_df["product"].tolist(),
        "positive": summary_df["positive"].tolist(),
        "neutral": summary_df["neutral"].tolist(),
        "negative": summary_df["negative"].tolist(),
    }


def safe_cell(value):
    """Excel formula injection se bachne ke liye."""
    s = str(value)
    if s and s[0] in "=+-@":
        return "'" + s
    return s


@app.errorhandler(413)
def too_large(e):
    return render_template(
        "index.html", active_tab="csv", csv_error="File bohat bari hai (max 1 MB)."
    ), 413


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = {
        "active_tab": "single",
        "review": "",
        "single_result": None,
        "single_error": None,
        "csv_error": None,
        "warning": None,
        "kpis": None,
        "summary": None,
        "rows": None,
        "chart": None,
    }

    if request.method == "POST":
        mode = request.form.get("mode", "single")

        if mode == "csv":
            ctx["active_tab"] = "csv"
            try:
                df = load_csv(request.files.get("file"))
                results = analyze_batch(df["feedback"].tolist())
                out = add_results(df, results)
                kpis = compute_kpis(out)
                if kpis["analysed"] == 0:
                    raise AIError("Koi review analyse nahi ho saka. Internet ya quota check kar ke dobara koshish karein.")
                if kpis["unknown"] > 0:
                    ctx["warning"] = f"{kpis['unknown']} reviews analyse nahi ho sake (unknown). Dobara upload kar ke dekh sakte hain."
                LAST["df"] = out
                summary_df = product_summary(out)
                ctx["kpis"] = kpis
                ctx["summary"] = summary_df.to_dict("records")
                ctx["chart"] = build_chart_data(kpis, summary_df)
                ctx["rows"] = out.to_dict("records")
            except (ValueError, AIError) as e:
                ctx["csv_error"] = str(e)
        else:
            ctx["review"] = request.form.get("review", "")
            try:
                ctx["single_result"] = analyze_review(ctx["review"])
            except (ValueError, AIError) as e:
                ctx["single_error"] = str(e)

    return render_template("index.html", **ctx)


@app.route("/single")
def single():
    return redirect(url_for("index"))


@app.route("/download")
def download():
    df = LAST["df"]
    if df is None:
        return redirect(url_for("index"))
    out = df.copy()
    for col in ["customer_id", "product", "feedback", "key_phrases"]:
        out[col] = out[col].map(safe_cell)
    csv_text = "\ufeff" + out.to_csv(index=False)
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=analysed_reviews.csv"},
    )


if __name__ == "__main__":
    app.run(debug=True)
