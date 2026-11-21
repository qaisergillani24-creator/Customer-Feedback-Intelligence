import os

from dotenv import load_dotenv
from google import genai

# .env file se GEMINI_API_KEY aur GEMINI_MODEL padhna
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL")

if not api_key or not model:
    raise SystemExit("GEMINI_API_KEY ya GEMINI_MODEL .env mein nahi mila.")

client = genai.Client(api_key=api_key)

review = "The delivery was late and the box was damaged, but support was helpful."

prompt = (
    "Classify the sentiment of this customer review as exactly one word: "
    "positive, negative, or neutral.\n"
    f"Review: {review}"
)

try:
    response = client.models.generate_content(model=model, contents=prompt)
    print("Review:   ", review)
    print("Sentiment:", response.text.strip())
except Exception as e:
    # Sirf error ki qism dikhate hain, key kabhi print nahi hoti
    print("Gemini call failed:", type(e).__name__)
    print(str(e)[:300])