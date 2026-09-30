from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "model", "fake_news_model.pkl")
)

vectorizer = joblib.load(
    os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")
)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@app.route("/")
def home():
    return jsonify({
        "message": "Fake News Detection API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    title = data.get("title", "")
    article = data.get("article", "")

    if not article.strip():
        return jsonify({
            "error": "Please enter an article"
        }), 400

    content = clean_text(title + " " + article)

    vector = vectorizer.transform([content])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    confidence = max(probabilities) * 100

    result = "REAL" if prediction == 1 else "FAKE"

    return jsonify({
        "result": result,
        "confidence": round(confidence, 2)
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )