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
