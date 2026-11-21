import pandas as pd

SENTIMENTS = ["positive", "neutral", "negative"]


def add_results(df, results):
    """Original DataFrame mein sentiment, confidence aur key_phrases ke columns jodta hai."""
    out = df.copy()
    out["sentiment"] = [r["sentiment"] for r in results]
    out["confidence"] = [round(r["confidence"], 2) for r in results]
    out["key_phrases"] = ["; ".join(r["key_phrases"]) for r in results]
    return out


def compute_kpis(df):
    """Poore dataset ke KPIs. Percentage sirf analyse hue reviews par nikalte hain."""
    total = len(df)
    known = df[df["sentiment"] != "unknown"]
    analysed = len(known)
    counts = {s: int((known["sentiment"] == s).sum()) for s in SENTIMENTS}

    kpis = {
        "total": total,
        "analysed": analysed,
        "unknown": total - analysed,
    }
    for s in SENTIMENTS:
        kpis[s] = counts[s]
    for s in SENTIMENTS:
        kpis[s + "_pct"] = round(counts[s] / analysed * 100, 1) if analysed else 0.0
    kpis["avg_confidence"] = round(float(known["confidence"].mean()), 2) if analysed else 0.0
    return kpis


def product_summary(df):
    """Har product ka summary, positive percentage ke hisaab se upar se neechay."""
    rows = []
    for product, group in df.groupby("product"):
        known = group[group["sentiment"] != "unknown"]
        n = len(known)
        row = {"product": product, "reviews": len(group)}
        for s in SENTIMENTS:
            row[s] = int((known["sentiment"] == s).sum())
        row["positive_pct"] = round(row["positive"] / n * 100, 1) if n else 0.0
        row["negative_pct"] = round(row["negative"] / n * 100, 1) if n else 0.0
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values("positive_pct", ascending=False).reset_index(drop=True)


# ---------------- V3: complaints aur product performance ----------------
from collections import Counter


def add_categories(df, negative_positions, categories):
    """Negative rows mein complaint_category bharta hai, baqi rows mein khali."""
    out = df.copy()
    out["complaint_category"] = ""
    for pos, cat in zip(negative_positions, categories):
        out.loc[pos, "complaint_category"] = cat
    return out


def complaint_summary(df):
    """Negative reviews ki categories ki ginti aur percentage."""
    if "complaint_category" not in df.columns:
        return pd.DataFrame(columns=["category", "count", "pct"])
    neg = df[(df["sentiment"] == "negative") & (df["complaint_category"] != "")]
    if neg.empty:
        return pd.DataFrame(columns=["category", "count", "pct"])
    counts = neg["complaint_category"].value_counts()
    total = int(counts.sum())
    out = pd.DataFrame({
        "category": counts.index.tolist(),
        "count": counts.values.tolist(),
    })
    out["pct"] = (out["count"] / total * 100).round(1)
    return out


def _status(net):
    if net >= 30:
        return "Strong"
    if net >= 0:
        return "Stable"
    return "At risk"


def product_performance(df):
    """Har product ka net score, rank, status aur sab se badi complaint."""
    s = product_summary(df)
    if s.empty:
        return s
    analysed = s["positive"] + s["neutral"] + s["negative"]
    net = ((s["positive"] - s["negative"]) / analysed.where(analysed > 0) * 100).round(1)
    s["net_score"] = net.fillna(0.0)
    s["status"] = s["net_score"].map(_status)

    top = {}
    if "complaint_category" in df.columns:
        neg = df[(df["sentiment"] == "negative") & (df["complaint_category"] != "")]
        for product, group in neg.groupby("product"):
            top[product] = group["complaint_category"].value_counts().idxmax()
    s["top_complaint"] = s["product"].map(top).fillna("-")

    s = s.sort_values(["net_score", "product"], ascending=[False, True]).reset_index(drop=True)
    s.insert(0, "rank", range(1, len(s) + 1))
    return s


def top_negative_phrases(df, n=8):
    """Negative reviews ke sab se aam key phrases."""
    neg = df[df["sentiment"] == "negative"]
    counter = Counter()
    for text in neg["key_phrases"]:
        for phrase in str(text).split(";"):
            phrase = phrase.strip().lower()
            if phrase:
                counter[phrase] += 1
    return [{"phrase": p, "count": c} for p, c in counter.most_common(n)]


def build_report_payload(kpis, perf_df, comp_df, phrases):
    """Gemini ko bhejne ke liye sirf summary numbers (koi customer_id ya review text nahi)."""
    keys = ["product", "reviews", "positive", "neutral", "negative",
            "net_score", "status", "top_complaint"]
    products = [{k: r[k] for k in keys} for r in perf_df.to_dict("records")]
    return {
        "total_reviews": kpis["total"],
        "analysed_reviews": kpis["analysed"],
        "positive_pct": kpis["positive_pct"],
        "neutral_pct": kpis["neutral_pct"],
        "negative_pct": kpis["negative_pct"],
        "products": products,
        "complaint_categories": comp_df.to_dict("records"),
        "top_negative_phrases": phrases,
        "note": "net_score = (positive - negative) / analysed reviews * 100",
    }
