import pandas as pd
from flask import Flask, Response, redirect, render_template, request, url_for

from file_loader import load_any

from ai_client import (
    AIError,
    analyze_batch,
    analyze_review,
    categorize_complaints,
    generate_report,
)
from analytics import (
    add_categories,
    add_results,
    build_report_payload,
    complaint_summary,
    compute_kpis,
    product_performance,
    top_negative_phrases,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024  # 6 MB
MAX_ROWS = 200
REQUIRED = ["customer_id", "product", "feedback"]

# Aakhri analysis memory mein (download aur report ke liye)
LAST = {
    "df": None, "kpis": None, "perf": None, "comp": None,
    "phrases": None, "report": None,
}


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


def safe_cell(value):
    """Excel formula injection se bachne ke liye."""
    s = str(value)
    if s and s[0] in "=+-@":
        return "'" + s
    return s


def results_ctx():
    """Aakhri analysis se template ke liye data."""
    if LAST["df"] is None:
        return {}
    perf = LAST["perf"]
    comp = LAST["comp"]
    kpis = LAST["kpis"]
    chart = {
        "sentiment": [kpis["positive"], kpis["neutral"], kpis["negative"]],
        "products": perf["product"].tolist(),
        "positive": perf["positive"].tolist(),
        "neutral": perf["neutral"].tolist(),
        "negative": perf["negative"].tolist(),
        "complaint_labels": comp["category"].tolist(),
        "complaint_counts": comp["count"].tolist(),
    }
    return {
        "kpis": kpis,
        "perf": perf.to_dict("records"),
        "comp": comp.to_dict("records"),
        "rows": LAST["df"].to_dict("records"),
        "chart": chart,
        "report": LAST["report"],
    }


@app.errorhandler(413)
def too_large(e):
    return render_template(
        "index.html", active_tab="csv", csv_error="File bohat bari hai (max 5 MB)."
    ), 413


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = {"active_tab": "single", "review": ""}

    if request.method == "POST":
        mode = request.form.get("mode", "single")

        if mode == "csv":
            ctx["active_tab"] = "csv"
            try:
                df, info = load_any(request.files.get("file"), request.form.get("default_product", ""))
                ctx["file_info"] = info
                results = analyze_batch(df["feedback"].tolist())
                out = add_results(df, results)
                kpis = compute_kpis(out)
                if kpis["analysed"] == 0:
                    raise AIError("Koi review analyse nahi ho saka. Internet ya quota check kar ke dobara koshish karein.")

                neg_positions = out.index[out["sentiment"] == "negative"].tolist()
                cats = categorize_complaints(out.loc[neg_positions, "feedback"].tolist()) if neg_positions else []
                out = add_categories(out, neg_positions, cats)

                warnings = []
                if kpis["unknown"] > 0:
                    warnings.append(f"{kpis['unknown']} reviews analyse nahi ho sake (unknown).")
                n_unc = sum(1 for c in cats if c == "uncategorized")
                if n_unc:
                    warnings.append(f"{n_unc} complaints ki category tay nahi ho saki (uncategorized).")
                if warnings:
                    ctx["warning"] = " ".join(warnings) + " Dobara upload kar ke dekh sakte hain."

                LAST["df"] = out
                LAST["kpis"] = kpis
                LAST["perf"] = product_performance(out)
                LAST["comp"] = complaint_summary(out)
                LAST["phrases"] = top_negative_phrases(out)
                LAST["report"] = None
                ctx.update(results_ctx())
            except (ValueError, AIError) as e:
                ctx["csv_error"] = str(e)
        else:
            ctx["review"] = request.form.get("review", "")
            try:
                ctx["single_result"] = analyze_review(ctx["review"])
            except (ValueError, AIError) as e:
                ctx["single_error"] = str(e)

    return render_template("index.html", **ctx)


@app.route("/report", methods=["POST"])
def report():
    if LAST["df"] is None:
        return redirect(url_for("index"))
    ctx = {"active_tab": "csv", "review": ""}
    try:
        payload = build_report_payload(
            LAST["kpis"], LAST["perf"], LAST["comp"], LAST["phrases"]
        )
        LAST["report"] = generate_report(payload)
    except AIError as e:
        ctx["report_error"] = str(e)
    ctx.update(results_ctx())
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
    for col in ["customer_id", "product", "feedback", "key_phrases", "complaint_category"]:
        out[col] = out[col].map(safe_cell)
    csv_text = "\ufeff" + out.to_csv(index=False)
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=analysed_reviews.csv"},
    )


if __name__ == "__main__":
    app.run(debug=True)
