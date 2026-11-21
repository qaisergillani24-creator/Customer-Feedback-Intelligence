import csv
import io
import json

import app as m
from ai_client import AIError

captured = {}


def fake_batch(texts):
    labels = ["positive", "neutral", "negative"]
    return [
        {"sentiment": labels[i % 3], "confidence": 0.9, "key_phrases": ["late delivery"]}
        for i, _ in enumerate(texts)
    ]


def fake_single(text):
    if not (text or "").strip():
        raise ValueError("Review khali hai. Pehle review likhein.")
    return {"sentiment": "positive", "confidence": 0.9, "key_phrases": ["great battery"]}


def fake_cats(texts):
    return ["delivery" for _ in texts]


def fake_report(payload):
    captured["payload"] = payload
    return {
        "summary": "Fake summary text",
        "key_findings": ["Finding one"],
        "recommendations": [
            {"priority": "High", "product": "Smartphone X", "action": "Fix delivery", "evidence": "8 negative reviews"}
        ],
    }


def failing_report(payload):
    raise AIError("Gemini request nakam hui (TestError).")


m.analyze_batch = fake_batch
m.analyze_review = fake_single
m.categorize_complaints = fake_cats
m.generate_report = fake_report
client = m.app.test_client()
failed = []


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        failed.append(name)


def upload(data, name):
    return client.post(
        "/",
        data={"mode": "csv", "file": (io.BytesIO(data), name)},
        content_type="multipart/form-data",
    )


r = client.get("/")
check("Home page opens (200)", r.status_code == 200)
check("Both tabs present", b"Single review" in r.data and b"CSV batch analysis" in r.data)

r = client.post("/", data={"mode": "single", "review": ""})
check("Single: empty review shows error", b"Review khali hai" in r.data)
r = client.post("/", data={"mode": "single", "review": "I love this phone"})
check("Single: result shown", b"text-bg-success" in r.data and b"great battery" in r.data)

r = client.post("/report")
check("Report before any upload redirects", r.status_code == 302)

sample = open("sample_reviews.csv", "rb").read()
r = upload(sample, "sample_reviews.csv")
check("CSV: upload OK (200)", r.status_code == 200)
check("CSV: KPI cards shown", b"Total reviews" in r.data)
check("CSV: three charts present",
      b'id="pie"' in r.data and b'id="bar"' in r.data and b'id="comp"' in r.data)
check("CSV: product performance table", b"Net score" in r.data and b"Top complaint" in r.data)
check("CSV: complaint breakdown shown", b"Complaint breakdown" in r.data and b"delivery" in r.data)
check("CSV: report button shown", b"Generate executive report" in r.data)

r = client.post("/report")
check("Report: text and recommendation shown",
      r.status_code == 200 and b"Fake summary text" in r.data and b"Fix delivery" in r.data)
check("Report: dashboard still shown", b"Total reviews" in r.data)
text = json.dumps(captured["payload"])
check("Report payload: no customer ids or review text",
      "customer_id" not in text and "C001" not in text and "Battery lasts" not in text)

m.generate_report = failing_report
r = client.post("/report")
check("Report failure shows clear error, dashboard kept",
      b"Gemini request nakam hui" in r.data and b"Total reviews" in r.data)

r = client.get("/download")
rows = list(csv.reader(io.StringIO(r.data.decode("utf-8-sig"))))
check("Download: CSV file", r.status_code == 200 and "text/csv" in r.headers["Content-Type"])
check("Download: 24 rows + header", len(rows) == 25)
check("Download: 7 columns incl. complaint_category",
      rows[0] == ["customer_id", "product", "feedback", "sentiment", "confidence",
                  "key_phrases", "complaint_category"])

r = upload(b"id,text\nA,hello\n", "bad_columns.csv")
check("Bad columns: clear error", b"columns missing" in r.data)
r = upload(b"", "empty.csv")
check("Empty file: clear error", b"khali" in r.data)
r = upload(b"customer_id,product,feedback\nA,P,hello\n", "notcsv.txt")
check("Wrong extension: clear error", b"Sirf .csv" in r.data)
r = client.post("/", data={"mode": "csv"})
check("No file chosen: clear error", b"Pehle aik CSV" in r.data)
r = client.get("/single")
check("/single redirects to dashboard", r.status_code == 302)

print()
print("ALL PASSED" if not failed else "FAILED: " + ", ".join(failed))
