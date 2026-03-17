"""
app.py
------
Flask web application for the Prompt Quality Classification System.
Serves the UI and provides a REST API endpoint for predictions.

Run: python app.py
Then open: http://127.0.0.1:5000
"""

import os
import joblib
from flask import Flask, request, jsonify, render_template

# ---------------------------------------------------------------------------
# Paths — resolve relative to this file so it works from any working directory
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "prompt_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(
        "Trained model not found. Please run 'python train_model.py' first."
    )

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# ---------------------------------------------------------------------------
# Flask app — use absolute template/static paths so it works from any cwd
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)


@app.route("/")
def home():
    """Serve the main UI page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Predict prompt quality via JSON API."""
    data = request.get_json(silent=True)
    if not data or "prompt" not in data:
        return jsonify({"error": "JSON body with 'prompt' key is required."}), 400

    prompt_text = str(data["prompt"]).strip()
    if not prompt_text:
        return jsonify({"error": "Prompt cannot be empty."}), 400

    # Vectorize and predict
    vector = vectorizer.transform([prompt_text])
    prediction = int(model.predict(vector)[0])

    # Confidence score (if model supports predict_proba)
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vector)[0]
        confidence = round(float(max(proba)) * 100, 1)

    quality = "High Quality" if prediction == 1 else "Low Quality"

    # Compute quick metrics for the prompt
    word_count = len(prompt_text.split())
    has_examples = any(
        kw in prompt_text.lower()
        for kw in ["example", "examples", "instance", "e.g."]
    )
    has_structure = any(
        kw in prompt_text.lower()
        for kw in ["bullet", "step", "list", "section", "heading", "format",
                    "structured", "numbered", "1)", "1."]
    )
    has_constraints = any(
        kw in prompt_text.lower()
        for kw in ["limit", "words", "sentences", "paragraph", "maximum",
                    "at least", "no more", "under "]
    )
    has_role = any(
        kw in prompt_text.lower()
        for kw in ["you are", "act as", "imagine you", "as a", "role"]
    )

    tips = []
    if word_count < 10:
        tips.append("Make your prompt longer and more specific.")
    if not has_examples:
        tips.append("Ask for examples to get richer output.")
    if not has_structure:
        tips.append("Request a structured format (bullet points, steps, headings).")
    if not has_constraints:
        tips.append("Add constraints like word limits or paragraph count.")
    if not has_role:
        tips.append("Assign a role to the AI (e.g., 'You are an expert teacher').")

    return jsonify({
        "prompt": prompt_text,
        "prediction": quality,
        "label": prediction,
        "confidence": confidence,
        "analysis": {
            "word_count": word_count,
            "has_examples": has_examples,
            "has_structure": has_structure,
            "has_constraints": has_constraints,
            "has_role": has_role,
        },
        "tips": tips,
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model_loaded": True})


if __name__ == "__main__":
    print("=" * 50)
    print("  Prompt Quality API")
    print("  Open http://127.0.0.1:8000 in your browser")
    print("=" * 50)
    app.run(debug=True, host="127.0.0.1", port=8000)
