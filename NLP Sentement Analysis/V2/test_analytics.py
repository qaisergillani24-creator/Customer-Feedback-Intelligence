import pandas as pd

from analytics import add_results, compute_kpis, product_summary

df = pd.DataFrame({
    "customer_id": ["A1", "A2", "A3", "A4", "A5", "A6"],
    "product": ["P1", "P1", "P1", "P2", "P2", "P2"],
    "feedback": ["a", "b", "c", "d", "e", "f"],
})
results = [
    {"sentiment": "positive", "confidence": 0.9, "key_phrases": ["good"]},
    {"sentiment": "positive", "confidence": 0.8, "key_phrases": ["fast", "cheap"]},
    {"sentiment": "negative", "confidence": 0.7, "key_phrases": []},
    {"sentiment": "neutral", "confidence": 0.6, "key_phrases": ["ok"]},
    {"sentiment": "negative", "confidence": 0.9, "key_phrases": ["broken"]},
    {"sentiment": "unknown", "confidence": 0.0, "key_phrases": []},
]

out = add_results(df, results)
print(out.to_string(index=False))
print()
print(compute_kpis(out))
print()
print(product_summary(out).to_string(index=False))
