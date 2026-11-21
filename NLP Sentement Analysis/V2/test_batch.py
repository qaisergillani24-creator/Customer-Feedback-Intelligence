import time

import pandas as pd

from ai_client import analyze_batch

df = pd.read_csv("sample_reviews.csv")
start = time.time()
results = analyze_batch(df["feedback"].tolist())
print(f"Time: {time.time() - start:.1f} sec")

for (_, row), r in zip(df.iterrows(), results):
    phrases = ", ".join(r["key_phrases"])
    print(row["customer_id"], row["product"], "|", r["sentiment"], round(r["confidence"], 2), "|", phrases)

print(pd.Series([r["sentiment"] for r in results]).value_counts())
