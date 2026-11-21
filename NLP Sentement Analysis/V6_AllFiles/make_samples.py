import json
import os

import openpyxl
from docx import Document

os.makedirs("samples", exist_ok=True)
reviews = [
    "Battery lasts two days and the screen is bright.",
    "Delivery was late and the box was damaged.",
    "Customer support never replied to my emails.",
    "Great value for the price, very happy.",
    "The app keeps crashing after the update.",
    "It works, nothing special.",
]

with open("samples/reviews.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(reviews))
with open("samples/reviews.json", "w", encoding="utf-8") as f:
    json.dump([{"customer_id": f"J{i + 1:02d}", "product": "Smartphone X", "feedback": r}
               for i, r in enumerate(reviews)], f)
wb = openpyxl.Workbook()
ws = wb.active
ws.append(["customer_id", "product", "feedback"])
for i, r in enumerate(reviews):
    ws.append([f"X{i + 1:02d}", "Laptop Pro", r])
wb.save("samples/reviews.xlsx")
d = Document()
for r in reviews:
    d.add_paragraph(r)
d.save("samples/reviews.docx")
with open("samples/reviews.html", "w", encoding="utf-8") as f:
    f.write("<html><body>" + "".join(f"<p>{r}</p>" for r in reviews) + "</body></html>")
print("samples ready:", sorted(os.listdir("samples")))
