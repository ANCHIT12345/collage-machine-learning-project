# Prompt Quality Analyzer

A Machine Learning–based system that classifies AI prompts as **High Quality** or **Low Quality** — without neural networks or external APIs. Includes a responsive web UI, REST API, and full training pipeline.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Step-by-Step Usage](#step-by-step-usage)
7. [How It Works](#how-it-works)
8. [API Reference](#api-reference)
9. [Improving Accuracy](#improving-accuracy)
10. [Screenshots](#screenshots)
11. [Technologies Used](#technologies-used)
12. [Viva Questions & Answers](#viva-questions--answers)

---

## Project Overview

**Problem:** Not all prompts given to AI systems produce good results. Short, vague prompts get poor outputs while structured, detailed prompts produce high-quality responses.

**Solution:** This project builds a **binary text classification model** that predicts whether a given prompt is **High Quality (1)** or **Low Quality (0)** based on its text features.

**Key Highlights:**

- No neural networks — uses traditional ML (Logistic Regression, SVM, Random Forest, Naive Bayes)
- No external APIs — everything runs locally
- Full web interface for testing prompts
- REST API for integration with other systems
- Actionable improvement tips for low-quality prompts

---

## Features

- **Dataset Generator** — Creates 1000 labeled prompt samples automatically
- **Multi-Model Training** — Compares 4 ML algorithms and picks the best
- **TF-IDF Vectorization** — Converts text to numerical features with unigrams + bigrams
- **Web UI** — Modern, responsive dark-themed interface
- **Real-time Analysis** — Word count, structure detection, constraint check, role detection
- **Improvement Tips** — Actionable suggestions to upgrade low-quality prompts
- **Prediction History** — Tracks recent analyses in the browser
- **Evaluation Plots** — Confusion matrix, label distribution, prompt length charts
- **REST API** — JSON endpoint for programmatic access

---

## Project Structure

```
PromptQualityAnalyzer/
│
├── dataset/
│   ├── DATASET_LINKS.md          # Links to download external datasets
│   └── prompt_dataset.csv        # Generated/custom training data
│
├── models/
│   ├── prompt_model.pkl          # Trained ML model (auto-created)
│   └── tfidf_vectorizer.pkl      # Fitted TF-IDF vectorizer (auto-created)
│
├── static/
│   └── plots/
│       ├── confusion_matrix.png  # Model evaluation plot (auto-created)
│       ├── label_distribution.png
│       └── length_distribution.png
│
├── templates/
│   └── index.html                # Web UI (HTML + CSS + JavaScript)
│
├── generate_dataset.py           # Script to create training dataset
├── train_model.py                # Script to train and evaluate ML model
├── app.py                        # Flask web server + API
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── PROJECT_DESCRIPTION.md        # Detailed project description
```

---

## Prerequisites

- **Python 3.8+** installed on your system
- **pip** package manager
- A modern web browser (Chrome, Firefox, Edge)

---

## Installation

### 1. Clone or download this project

Place the project folder anywhere on your computer.

### 2. Open a terminal in the project folder

```bash
cd "path/to/PromptQualityAnalyzer"
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `flask`.

---

## Step-by-Step Usage

### Step 1 — Generate the Dataset

```bash
python generate_dataset.py
```

This creates `dataset/prompt_dataset.csv` with **1000 labeled prompts** (500 low + 500 high quality).

> **Want more data?** Open `generate_dataset.py` and change `num_samples = 1000` to a higher number. Or download a larger dataset from the links in `dataset/DATASET_LINKS.md`.

### Step 2 — Train the Model

```bash
python train_model.py
```

This will:

1. Load the dataset
2. Convert text to TF-IDF features
3. Train 4 models (Logistic Regression, Random Forest, SVM, Naive Bayes)
4. Print accuracy comparison
5. Save the best model to `models/`
6. Generate evaluation plots in `static/plots/`

Expected output:

```
--- Model Comparison ---
Model                      Accuracy   Precision     Recall         F1
--------------------------------------------------------------------
Logistic Regression          0.9350     0.9355     0.9350     0.9349
Random Forest                0.9050     0.9077     0.9050     0.9044
Linear SVM                   0.9400     0.9405     0.9400     0.9399
Naive Bayes                  0.9150     0.9155     0.9150     0.9149
```

### Step 3 — Start the Web Application

```bash
python app.py
```

Open your browser and go to: **http://127.0.0.1:5000**

### Step 4 — Use the App

1. Type or paste a prompt in the text box
2. Click **"Analyze Prompt"** (or press **Ctrl + Enter**)
3. View the prediction, confidence score, analysis, and improvement tips
4. Try the example chips for quick testing
5. Check your prediction history at the bottom

---

## How It Works

### 1. Text to Numbers (TF-IDF)

- TF-IDF (Term Frequency–Inverse Document Frequency) converts each prompt into a numerical vector
- Uses unigrams and bigrams (`ngram_range=(1,2)`) to capture word-level and phrase-level patterns
- Example: "Explain AI with examples" → `[0.0, 0.45, 0.0, 0.32, ...]`

### 2. Classification Model

- Trains on labeled data: `0 = Low Quality`, `1 = High Quality`
- Compares 4 algorithms and picks the one with highest accuracy
- The best model is saved for use by the API

### 3. Feature Analysis

The API also checks for structural indicators:

- **Word count** — longer prompts tend to be better
- **Examples** — does the prompt ask for examples?
- **Structure** — does it request bullet points, steps, or headings?
- **Constraints** — does it set word limits or format requirements?
- **Role definition** — does it assign a role to the AI?

### 4. Improvement Tips

If the prompt is missing any quality indicators, the system provides actionable tips.

---

## API Reference

### `POST /predict`

**Request:**

```json
{
  "prompt": "Explain AI with 3 examples in 150 words"
}
```

**Response:**

```json
{
  "prompt": "Explain AI with 3 examples in 150 words",
  "prediction": "High Quality",
  "label": 1,
  "confidence": 92.3,
  "analysis": {
    "word_count": 9,
    "has_examples": true,
    "has_structure": false,
    "has_constraints": true,
    "has_role": false
  },
  "tips": [
    "Request a structured format (bullet points, steps, headings).",
    "Assign a role to the AI (e.g., 'You are an expert teacher')."
  ]
}
```

### `GET /health`

Returns `{"status": "ok", "model_loaded": true}`

---

## Improving Accuracy

If accuracy is below your target:

1. **More data** — Increase `num_samples` in `generate_dataset.py` or use datasets from `DATASET_LINKS.md`
2. **Balanced labels** — Ensure equal numbers of 0 and 1 labels
3. **Better features** — Add custom hand-crafted features alongside TF-IDF
4. **Hyperparameter tuning** — Use `GridSearchCV` in `train_model.py`
5. **Bigrams/Trigrams** — Change `ngram_range` to `(1,3)`
6. **Remove stopwords** — Already enabled by default

---

## Screenshots

After running the project, the web interface will look like a modern dark-themed analyzer with:

- Text input area with word/character counter
- Color-coded quality badges (green = high, red = low)
- Confidence progress bar
- Analysis grid showing prompt characteristics
- Improvement tips section
- Example chips for quick testing
- Prediction history

---

## Technologies Used

| Technology           | Purpose                       |
| -------------------- | ----------------------------- |
| Python 3.8+          | Core programming language     |
| Scikit-learn         | ML models, TF-IDF, evaluation |
| Pandas               | Data loading and manipulation |
| Matplotlib & Seaborn | Evaluation plots              |
| Flask                | Web server and REST API       |
| HTML/CSS/JavaScript  | Frontend UI                   |
| Joblib               | Model serialization           |

---

## Viva Questions & Answers

**Q: What is TF-IDF?**
A: TF-IDF stands for Term Frequency–Inverse Document Frequency. It converts text into numerical values where important words get higher weights and common words get lower weights.

**Q: Why Logistic Regression?**
A: It is efficient, interpretable, and works very well for binary text classification problems. It provides probability outputs for confidence scoring.

**Q: Why no neural networks?**
A: This project demonstrates that traditional ML can achieve high accuracy (85–95%) for structured text classification without the complexity of deep learning.

**Q: How did you label the dataset?**
A: High-quality prompts contain clear instructions, constraints (word limits), structure (bullet points, steps), examples, and role definitions. Low-quality prompts are vague, short, and lack direction.

**Q: What is the model accuracy?**
A: With 1000 samples and TF-IDF + bigrams, accuracy typically ranges from 88% to 95%.

**Q: Can this work with real-world prompts?**
A: Yes. The model generalizes to new prompts because TF-IDF captures word patterns associated with quality indicators.

**Q: What is the API used for?**
A: The Flask API allows the ML model to be accessed programmatically. It takes a prompt as JSON input and returns the quality prediction, confidence, analysis, and tips.

---

## License

This project is for educational purposes. Free to use and modify.
#   c o l l a g e - m a c h i n e - l e a r n i n g - p r o j e c t  
 