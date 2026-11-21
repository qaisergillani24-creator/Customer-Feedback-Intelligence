import csv
import io
import json
import zipfile

from werkzeug.datastructures import FileStorage

import app as m
import file_loader as fl

failed = []


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        failed.append(name)


def FS(name, data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return FileStorage(stream=io.BytesIO(data), filename=name)


def load(name, data, product=""):
    return fl.load_any(FS(name, data), product)


def fails(name, data, product=""):
    try:
        load(name, data, product)
    except fl.FileError as e:
        return str(e)
    except Exception as e:
        return "WRONG:" + type(e).__name__
    return None


def clean_error(msg):
    return msg is not None and not msg.startswith("WRONG:")


# ---------- Table wali files ----------
df, info = load("a.csv", "customer_id,product,feedback\nC1,Phone,Great battery\nC2,Laptop,Too slow\n")
check("CSV standard", list(df.columns) == fl.REQUIRED and len(df) == 2
      and df.loc[0, "customer_id"] == "C1" and df.loc[1, "product"] == "Laptop" and info["kind"] == "csv")

df, _ = load("b.csv", "ID,Product Name,Review\n7,Tablet,Screen is dim\n")
check("CSV with different column names", df.loc[0, "customer_id"] == "7"
      and df.loc[0, "product"] == "Tablet" and df.loc[0, "feedback"] == "Screen is dim")

df, _ = load("c.csv", "id,text\nA,hello world thing\n")
check("CSV with a text column and no product", df.loc[0, "feedback"] == "hello world thing"
      and df.loc[0, "product"] == "Unknown")

df, info = load("d.csv", "name,long_opinion\nBob,This product is really great\n")
check("Unknown text column is detected and noted", df.loc[0, "feedback"] == "This product is really great"
      and any("samjha gaya" in n for n in info["notes"]))
check("CSV with no text column is rejected", clean_error(fails("e.csv", "a,b\n1,2\n")))

df, _ = load("f.csv", "feedback\nGreat, really great\nBad, very bad\n")
check("Single column with commas still works", len(df) == 2 and df.loc[0, "feedback"] == "Great, really great")

df, _ = load("g.tsv", "customer_id\tproduct\tfeedback\nT1\tPhone\tNice one\n")
check("TSV works", len(df) == 1 and df.loc[0, "product"] == "Phone")

import openpyxl

wb = openpyxl.Workbook()
ws = wb.active
ws.append(["customer_id", "product", "feedback"])
ws.append(["X1", "Phone", "Nice screen and fast"])
ws.append(["X2", "Phone", "Battery dies quickly"])
buf = io.BytesIO()
wb.save(buf)
df, info = load("h.xlsx", buf.getvalue())
check("Excel works", len(df) == 2 and df.loc[0, "product"] == "Phone" and info["kind"] == "xlsx")

df, _ = load("i.json", json.dumps([{"customer_id": "J1", "product": "P", "feedback": "good one"}]))
check("JSON list of records", len(df) == 1 and df.loc[0, "customer_id"] == "J1")
df, _ = load("j.json", json.dumps({"reviews": [{"review": "fine product here", "product_name": "Q"}]}))
check("JSON object with a reviews list", len(df) == 1 and df.loc[0, "product"] == "Q")
df, _ = load("k.json", json.dumps(["first review", "second review"]))
check("JSON list of strings", len(df) == 2)
check("Broken JSON is rejected", clean_error(fails("l.json", "{not json")))

# ---------- Text wali files ----------
df, _ = load("m.txt", "First review here.\n\nSecond review here.\n\n3. Third review here.\n", "My Product")
check("Text paragraphs become reviews, product is used",
      len(df) == 3 and df.loc[2, "feedback"] == "Third review here." and df.loc[0, "product"] == "My Product")
df, _ = load("n.txt", "Line one text\nLine two text\n")
check("Text lines become reviews", len(df) == 2)
df, _ = load("o.txt", "- bullet one\n- bullet two\n")
check("Bullets are removed", df.loc[0, "feedback"] == "bullet one")
check("Tiny text is rejected", clean_error(fails("p.txt", "ok")))
df, _ = load("q.txt", "customer_id\tproduct\tfeedback\nT1\tPhone\tNice\n")
check("Tab-separated table inside a txt file", len(df) == 1 and df.loc[0, "product"] == "Phone")

df, info = load("r.dat", "Review one is here\nReview two is here\n")
check("Unknown extension that is text is read", len(df) == 2 and any("anjaan" in n for n in info["notes"]))
check("Binary file is rejected", "parhi nahi" in (fails("s.bin", bytes([0, 1, 2, 3] * 200)) or ""))

from docx import Document

d = Document()
d.add_paragraph("Battery is great")
d.add_paragraph("Delivery was very late")
d.add_paragraph("Support was helpful")
buf = io.BytesIO()
d.save(buf)
df, _ = load("t.docx", buf.getvalue(), "Phone")
check("Word works", len(df) == 3 and df.loc[1, "feedback"] == "Delivery was very late" and df.loc[0, "product"] == "Phone")

from pptx import Presentation

prs = Presentation()
slide = prs.slides.add_slide(prs.slide_layouts[5])
slide.shapes.title.text = "Great product I love it"
tb = slide.shapes.add_textbox(0, 0, 100, 100)
tb.text_frame.text = "Slow delivery again"
buf = io.BytesIO()
prs.save(buf)
df, _ = load("u.pptx", buf.getvalue())
check("PowerPoint works", "Slow delivery again" in df["feedback"].tolist() and len(df) >= 2)

df, _ = load("v.html", "<html><script>x=1</script><p>Nice phone &amp; screen</p><p>Bad delivery</p></html>")
check("HTML works (scripts removed, entities decoded)",
      df["feedback"].tolist() == ["Nice phone & screen", "Bad delivery"])

# ---------- Image aur scanned (Gemini ke bina) ----------
png = b"\x89PNG\r\n\x1a\n" + b"0" * 50
fl._ocr = lambda data, mime: "Great product\n\nTerrible delivery"
df, info = load("w.png", png)
check("Image goes through OCR", len(df) == 2 and any("Gemini" in n for n in info["notes"]))
check("Fake PDF is rejected", clean_error(fails("x.pdf", "hello this is not a pdf")))
check("Fake PNG is rejected", clean_error(fails("y.png", "hello this is not a png")))

# ---------- Hifazat ----------
check("Too big is rejected", "bohat bari" in (fails("big.txt", b"a" * (fl.MAX_FILE_BYTES + 1)) or ""))
check("Empty file is rejected", "khali" in (fails("e.txt", b"  \n") or ""))
try:
    fl.load_any(None)
    nofile = False
except fl.FileError as e:
    nofile = "Pehle aik file" in str(e)
check("No file is rejected", nofile)
check("Old .xls is rejected with advice", "xlsx" in (fails("z.xls", b"data") or ""))

df, info = load("many.txt", "\n".join(f"Review number {i} is here" for i in range(250)))
check("More than 200 reviews are cut with a note",
      len(df) == fl.MAX_ROWS and any("250" in n for n in info["notes"]))

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as z:
    for i in range(fl.MAX_ZIP_ENTRIES + 1):
        z.writestr(f"f{i}.txt", "x")
check("Zip bomb style docx is rejected", "hissay" in (fails("bomb.docx", buf.getvalue()) or ""))

cp = fl._clean_product("<b>x</b>" + "y" * 100)
check("Product name is cleaned and capped", len(cp) <= 60 and "<" not in cp)

# ---------- Dashboard ke sath ----------
def fake_batch(texts):
    labels = ["positive", "neutral", "negative"]
    return [{"sentiment": labels[i % 3], "confidence": 0.9, "key_phrases": ["late delivery"]}
            for i, _ in enumerate(texts)]


def fake_cats(texts):
    return ["delivery" for _ in texts]


def fake_report(payload):
    return {"summary": "Fake summary text", "key_findings": ["Finding one"], "recommendations": []}


m.analyze_batch = fake_batch
m.categorize_complaints = fake_cats
m.generate_report = fake_report
m.analyze_review = lambda t: {"sentiment": "positive", "confidence": 0.9, "key_phrases": ["great"]}
client = m.app.test_client()


def upload(name, data, product=""):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return client.post(
        "/",
        data={"mode": "csv", "default_product": product, "file": (io.BytesIO(data), name)},
        content_type="multipart/form-data",
    )


home = client.get("/")
check("Home opens with the file tab and analyst link",
      home.status_code == 200 and b"File analysis" in home.data and b'href="/analyst"' in home.data)

r = upload("sample_reviews.csv", open("sample_reviews.csv", "rb").read())
check("CSV upload shows KPIs, charts and file info",
      r.status_code == 200 and b"Total reviews" in r.data and b"24 reviews found" in r.data
      and b'id="pie"' in r.data and b'id="comp"' in r.data)
rows = list(csv.reader(io.StringIO(client.get("/download").data.decode("utf-8-sig"))))
check("Download still gives 24 rows and 7 columns", len(rows) == 25 and len(rows[0]) == 7)
r = client.post("/report")
check("Executive report still works", b"Fake summary text" in r.data)

r = upload("notes.txt", "Battery is great\n\nDelivery was late\n\nSupport helped me", "My Product")
check("Text upload uses the product name", r.status_code == 200 and b"My Product" in r.data
      and b"3 reviews found" in r.data)
r = upload("data.json", json.dumps([{"review": "very nice item", "product": "Json Item"}]))
check("JSON upload works", b"Json Item" in r.data and b"1 reviews found" in r.data)
r = upload("tool.exe", bytes([0, 1, 2, 3] * 200))
check("Binary upload shows a clear error", b"parhi nahi" in r.data)
r = client.post("/", data={"mode": "csv"})
check("No file shows a clear error", b"Pehle aik file" in r.data)
r = upload("huge.txt", b"a" * (7 * 1024 * 1024))
check("Huge upload gives 413 with a clear message", r.status_code == 413 and b"max 5 MB" in r.data)
r = client.post("/", data={"mode": "single", "review": "I love it"})
check("Single review tab still works", b"text-bg-success" in r.data)
rules = {x.rule for x in m.app.url_map.iter_rules()}
check("Analyst routes still exist", {"/analyst", "/api/ask", "/api/clear", "/report"} <= rules)

print()
print("ALL PASSED" if not failed else "FAILED: " + ", ".join(failed))
