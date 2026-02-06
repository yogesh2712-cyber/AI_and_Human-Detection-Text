from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model & vectorizer
with open("aig_detector_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None

    if request.method == "POST":
        text = request.form["text"]

        if text.strip() != "":
            text_vector = vectorizer.transform([text])
            prob = model.predict_proba(text_vector)[0]
            result = model.predict(text_vector)[0]

            prediction = "AI Generated 🤖" if result == 1 else "Human Written 🧑"
            confidence = round(max(prob) * 100, 2)

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
