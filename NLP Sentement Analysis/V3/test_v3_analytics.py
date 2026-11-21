import json

import pandas as pd

from ai_client import AIError, COMPLAINT_CATEGORIES, _clean_report, categorize_complaints
from analytics import (
    add_categories,
    build_report_payload,
    complaint_summary,
    compute_kpis,
    product_performance,
    top_negative_phrases,
)

failed = []


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        failed.append(name)


df = pd.DataFrame({
    "customer_id": [f"C{i}" for i in range(9)],
    "product": ["P1", "P1", "P1", "P1", "P2", "P2", "P2", "P2", "P2"],
    "feedback": [f"secret text {i}" for i in range(9)],
    "sentiment": ["positive", "positive", "negative", "negative", "negative",
                  "negative", "negative", "neutral", "unknown"],
    "confidence": [0.9] * 9,
    "key_phrases": ["", "", "late delivery; box damaged", "late delivery",
                    "broken screen", "broken screen; poor quality",
                    "late delivery", "", ""],
})

out = add_categories(df, [2, 3, 4, 5, 6],
                     ["delivery", "delivery", "product quality", "product quality", "delivery"])
check("Non-negative rows have empty category",
      out.loc[0, "complaint_category"] == "" and out.loc[7, "complaint_category"] == "")

cs = complaint_summary(out)
check("Complaint summary: delivery 3 (60.0%), product quality 2 (40.0%)",
      cs.iloc[0]["category"] == "delivery" and int(cs.iloc[0]["count"]) == 3
      and cs.iloc[0]["pct"] == 60.0 and int(cs.iloc[1]["count"]) == 2)

perf = product_performance(out)
p1 = perf[perf["product"] == "P1"].iloc[0]
p2 = perf[perf["product"] == "P2"].iloc[0]
check("P1: rank 1, net 0.0, Stable, top complaint delivery",
      int(p1["rank"]) == 1 and p1["net_score"] == 0.0
      and p1["status"] == "Stable" and p1["top_complaint"] == "delivery")
check("P2: rank 2, net -75.0, At risk, top complaint product quality",
      int(p2["rank"]) == 2 and p2["net_score"] == -75.0
      and p2["status"] == "At risk" and p2["top_complaint"] == "product quality")

ph = top_negative_phrases(out)
check("Top phrases: late delivery 3, broken screen 2",
      ph[0] == {"phrase": "late delivery", "count": 3}
      and ph[1] == {"phrase": "broken screen", "count": 2})

payload = build_report_payload(compute_kpis(out), perf, cs, ph)
text = json.dumps(payload)
check("Report payload is valid JSON", len(text) > 50)
check("Payload has no customer ids or review text",
      "customer_id" not in text and "secret text" not in text and "C0" not in text)

good = _clean_report({
    "summary": "S",
    "key_findings": ["a", "", "b"],
    "recommendations": [
        {"priority": "high", "product": "P1", "action": "Fix X", "evidence": "50%"},
        {"action": ""},
        "bad",
    ],
})
check("Report cleaner keeps valid items only",
      good["summary"] == "S" and good["key_findings"] == ["a", "b"]
      and len(good["recommendations"]) == 1
      and good["recommendations"][0]["priority"] == "High")

try:
    _clean_report({"summary": ""})
    check("Empty report summary is rejected", False)
except AIError:
    check("Empty report summary is rejected", True)

check("categorize_complaints([]) returns []", categorize_complaints([]) == [])
check("Category list has delivery and other",
      "delivery" in COMPLAINT_CATEGORIES and "other" in COMPLAINT_CATEGORIES)

print()
print("ALL PASSED" if not failed else "FAILED: " + ", ".join(failed))
