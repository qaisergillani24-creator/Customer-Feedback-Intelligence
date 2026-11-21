from flask import Flask, render_template, request

from ai_client import AIError, analyze_review

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    review = ""
    result = None
    error = None

    if request.method == "POST":
        review = request.form.get("review", "")
        try:
            result = analyze_review(review)
        except (ValueError, AIError) as e:
            error = str(e)

    return render_template("index.html", review=review, result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True)