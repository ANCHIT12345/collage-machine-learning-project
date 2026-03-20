"""
app.py
------
Flask web application for the Prompt Quality Classification System.
Serves the UI and provides a REST API endpoint for predictions.

Run: python app.py
Then open: http://127.0.0.1:5000
"""

import os
import io
import csv
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

    # Compute a 0-100 quality score from analysis signals
    score = 0
    if word_count >= 10: score += 20
    elif word_count >= 5: score += 10
    if has_examples: score += 20
    if has_structure: score += 20
    if has_constraints: score += 20
    if has_role: score += 20

    return jsonify({
        "prompt": prompt_text,
        "prediction": quality,
        "label": prediction,
        "confidence": confidence,
        "score": score,
        "analysis": {
            "word_count": word_count,
            "has_examples": has_examples,
            "has_structure": has_structure,
            "has_constraints": has_constraints,
            "has_role": has_role,
        },
        "tips": tips,
    })


@app.route("/rewrite", methods=["POST"])
def rewrite():
    """Auto-improve a low-quality prompt using rule-based rewriting."""
    data = request.get_json(silent=True)
    if not data or "prompt" not in data:
        return jsonify({"error": "JSON body with 'prompt' key is required."}), 400

    prompt_text = str(data["prompt"]).strip()
    if not prompt_text:
        return jsonify({"error": "Prompt cannot be empty."}), 400

    rewritten = prompt_text

    # Add role if missing
    if not any(kw in prompt_text.lower() for kw in ["you are", "act as", "as a", "role"]):
        rewritten = "You are an expert. " + rewritten

    # Add structure if missing
    if not any(kw in rewritten.lower() for kw in ["bullet", "step", "list", "heading", "format", "structured", "numbered"]):
        rewritten += " Provide your answer in a structured format with clear headings."

    # Add examples if missing
    if not any(kw in rewritten.lower() for kw in ["example", "instance", "e.g."]):
        rewritten += " Include 2-3 real-world examples."

    # Add constraint if missing
    if not any(kw in rewritten.lower() for kw in ["limit", "words", "sentences", "paragraph", "maximum", "at least", "under "]):
        rewritten += " Keep your response under 200 words."

    return jsonify({"original": prompt_text, "rewritten": rewritten})


@app.route("/batch", methods=["POST"])
def batch():
    """Analyze multiple prompts from an uploaded CSV file."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send a CSV with a 'prompt' column."}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported."}), 400

    stream = io.StringIO(file.stream.read().decode("utf-8"))
    reader = csv.DictReader(stream)

    if "prompt" not in (reader.fieldnames or []):
        return jsonify({"error": "CSV must have a 'prompt' column."}), 400

    results = []
    for row in reader:
        p = str(row["prompt"]).strip()
        if not p:
            continue
        vec = vectorizer.transform([p])
        pred = int(model.predict(vec)[0])
        conf = None
        if hasattr(model, "predict_proba"):
            conf = round(float(max(model.predict_proba(vec)[0])) * 100, 1)
        results.append({
            "prompt": p,
            "prediction": "High Quality" if pred == 1 else "Low Quality",
            "label": pred,
            "confidence": conf,
        })

    return jsonify({"total": len(results), "results": results})


@app.route("/response", methods=["POST"])
def generate_response():
    """Generate a simulated rule-based AI response for a given prompt."""
    data = request.get_json(silent=True)
    if not data or "prompt" not in data:
        return jsonify({"error": "JSON body with 'prompt' key is required."}), 400

    prompt_text = str(data["prompt"]).strip()
    if not prompt_text:
        return jsonify({"error": "Prompt cannot be empty."}), 400

    p = prompt_text.lower()

    # Detect topic keywords
    topic_map = {
        "machine learning": "Machine Learning", "ml": "Machine Learning",
        "deep learning": "Deep Learning", "neural network": "Neural Networks",
        "artificial intelligence": "Artificial Intelligence", " ai ": "Artificial Intelligence",
        "python": "Python Programming", "javascript": "JavaScript",
        "blockchain": "Blockchain", "cloud": "Cloud Computing",
        "cybersecurity": "Cybersecurity", "data science": "Data Science",
        "nlp": "Natural Language Processing", "natural language": "Natural Language Processing",
        "computer vision": "Computer Vision", "api": "API Development",
        "docker": "Docker & Containers", "kubernetes": "Kubernetes",
        "database": "Database Management", "sql": "SQL & Databases",
        "climate": "Climate Change", "quantum": "Quantum Computing",
        "robotics": "Robotics", "iot": "Internet of Things",
    }
    topic = "the requested topic"
    for kw, label in topic_map.items():
        if kw in p:
            topic = label
            break

    has_role    = any(kw in p for kw in ["you are", "act as", "as a", "role"])
    has_steps   = any(kw in p for kw in ["step", "steps", "how to", "guide", "tutorial"])
    has_compare = any(kw in p for kw in ["compare", "difference", "vs", "versus", "contrast"])
    has_list    = any(kw in p for kw in ["list", "bullet", "enumerate", "advantages", "disadvantages"])
    has_explain = any(kw in p for kw in ["explain", "what is", "define", "describe", "overview"])
    has_example = any(kw in p for kw in ["example", "instance", "e.g.", "such as"])
    has_summary = any(kw in p for kw in ["summary", "summarize", "brief", "short", "concise"])

    role_intro = "As an expert in this field, " if has_role else ""

    if has_compare:
        response = (
            f"{role_intro}here is a comparison related to **{topic}**:\n\n"
            f"**Option A** focuses on foundational principles and is widely adopted in academic settings. "
            f"It offers strong theoretical backing but may require more setup time.\n\n"
            f"**Option B** is more practical and industry-oriented, offering faster results with less configuration. "
            f"However, it may sacrifice some depth of understanding.\n\n"
            f"| Aspect | Option A | Option B |\n"
            f"|--------|----------|----------|\n"
            f"| Learning Curve | Moderate | Low |\n"
            f"| Industry Use | High | Very High |\n"
            f"| Flexibility | High | Moderate |\n"
            f"| Community Support | Large | Growing |\n\n"
            f"**Recommendation:** Choose based on your goal — Option A for depth, Option B for speed."
        )
    elif has_steps:
        response = (
            f"{role_intro}here is a step-by-step guide on **{topic}**:\n\n"
            f"**Step 1 — Understand the Basics**\n"
            f"Begin by familiarizing yourself with the core concepts. Read introductory material and watch overview videos.\n\n"
            f"**Step 2 — Set Up Your Environment**\n"
            f"Install the necessary tools and dependencies. Ensure your system meets the prerequisites.\n\n"
            f"**Step 3 — Build a Simple Project**\n"
            f"Apply what you've learned by creating a small hands-on project. This reinforces understanding.\n\n"
            f"**Step 4 — Explore Advanced Concepts**\n"
            f"Once comfortable, dive into advanced topics, optimization techniques, and best practices.\n\n"
            f"**Step 5 — Review & Iterate**\n"
            f"Test your knowledge, seek feedback, and continuously improve your implementation."
        )
    elif has_list:
        response = (
            f"{role_intro}here are key points about **{topic}**:\n\n"
            f"• **Definition:** {topic} refers to a set of techniques and methodologies used to solve complex problems efficiently.\n"
            f"• **Core Principle:** It relies on data-driven decision making and iterative improvement.\n"
            f"• **Key Advantage 1:** Automates repetitive tasks, saving significant time and resources.\n"
            f"• **Key Advantage 2:** Scales effectively with increasing data and complexity.\n"
            f"• **Key Advantage 3:** Continuously improves performance through feedback loops.\n"
            f"• **Limitation:** Requires quality data and domain expertise to implement correctly.\n"
            f"• **Use Cases:** Healthcare diagnostics, financial forecasting, smart automation, and more."
        )
    elif has_summary:
        response = (
            f"{role_intro}here is a concise summary of **{topic}**:\n\n"
            f"{topic} is a transformative field that combines theoretical foundations with practical applications. "
            f"It enables systems to process information intelligently, adapt to new inputs, and deliver meaningful outputs. "
            f"Its impact spans industries including healthcare, finance, education, and engineering.\n\n"
            f"**Key Takeaway:** Mastering {topic} opens doors to solving real-world problems at scale."
        )
    else:
        response = (
            f"{role_intro}here is an explanation of **{topic}**:\n\n"
            f"**Overview**\n"
            f"{topic} is a rapidly evolving domain that sits at the intersection of theory and practice. "
            f"It provides tools and frameworks that allow practitioners to tackle complex, real-world challenges.\n\n"
            f"**How It Works**\n"
            f"At its core, {topic} operates by processing structured or unstructured inputs, applying learned or rule-based transformations, "
            f"and producing actionable outputs. The process is iterative and improves with more data and refinement.\n\n"
            f"**Real-World Applications**\n"
            f"1. Used in industry to automate and optimize workflows\n"
            f"2. Applied in research to accelerate discovery\n"
            f"3. Integrated into consumer products for smarter experiences\n\n"
            f"**Conclusion**\n"
            f"{topic} continues to grow in importance. Building a strong foundation now will prepare you for future advancements."
        )

    if has_example and "example" not in response.lower():
        response += (
            f"\n\n**Example**\n"
            f"Consider a real-world scenario: a company uses {topic} to reduce operational costs by 30% "
            f"within the first year of deployment by automating key decision-making processes."
        )

    return jsonify({"prompt": prompt_text, "response": response, "topic": topic})


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
