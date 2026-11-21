import csv
import io

import app as m


def fake_batch(texts):
    labels = ["positive", "neutral", "negative"]
    return [
        {"sentiment": labels[i % 3], "confidence": 0.9, "key_phrases": ["test phrase"]}
        for i, _ in enumerate(texts)
    ]


def fake_single(text):
    if not (text or "").strip():
        raise ValueError("Review khali hai. Pehle review likhein.")
    return {"sentiment": "positive", "confidence": 0.9, "key_phrases": ["great battery"]}


m.analyze_batch = fake_batch
m.analyze_review = fake_single
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

sample = open("sample_reviews.csv", "rb").read()
r = upload(sample, "sample_reviews.csv")
check("CSV: upload OK (200)", r.status_code == 200)
check("CSV: KPI cards shown", b"Total reviews" in r.data)
check("CSV: both charts present", b'id="pie"' in r.data and b'id="bar"' in r.data)
check("CSV: product summary shown", b"Smartphone X" in r.data and b"Smart Watch" in r.data)
check("CSV: download link shown", b"/download" in r.data)

r = client.get("/download")
rows = list(csv.reader(io.StringIO(r.data.decode("utf-8-sig"))))
check("Download: CSV file", r.status_code == 200 and "text/csv" in r.headers["Content-Type"])
check("Download: 24 rows + header", len(rows) == 25)
check(
    "Download: 6 columns",
    rows[0] == ["customer_id", "product", "feedback", "sentiment", "confidence", "key_phrases"],
)

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
