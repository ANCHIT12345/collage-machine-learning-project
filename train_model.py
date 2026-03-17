"""
train_model.py
--------------
Trains a Prompt Quality Classification model using TF-IDF + ML algorithms.
Compares multiple models and saves the best one.

Run: python train_model.py
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)


def load_data(path="dataset/prompt_dataset.csv"):
    """Load and validate the dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Please run 'python generate_dataset.py' first, or place your own "
            "prompt_dataset.csv inside the dataset/ folder."
        )
    df = pd.read_csv(path)
    if "prompt" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must have 'prompt' and 'label' columns.")
    df = df.dropna(subset=["prompt", "label"])
    df["label"] = df["label"].astype(int)
    return df


def explore_data(df):
    """Print dataset statistics."""
    print("\n--- Dataset Overview ---")
    print(f"Total samples : {len(df)}")
    print(f"Low  Quality  : {len(df[df['label'] == 0])}")
    print(f"High Quality  : {len(df[df['label'] == 1])}")
    print(f"\nSample prompts:")
    for _, row in df.head(5).iterrows():
        label_text = "HIGH" if row["label"] == 1 else "LOW"
        print(f"  [{label_text}] {row['prompt'][:80]}...")


def build_features(df):
    """Convert text to TF-IDF features."""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words="english",
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(df["prompt"])
    y = df["label"].values
    return X, y, vectorizer


def train_and_compare(X_train, X_test, y_train, y_test):
    """Train multiple models and return the best one."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Linear SVM": LinearSVC(max_iter=2000),
        "Naive Bayes": MultinomialNB(alpha=1.0),
    }

    results = {}
    best_model = None
    best_accuracy = 0
    best_name = ""

    print("\n--- Model Comparison ---")
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 68)

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        precision = report["weighted avg"]["precision"]
        recall = report["weighted avg"]["recall"]
        f1 = report["weighted avg"]["f1-score"]

        print(f"{name:<25} {acc:>10.4f} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")

        results[name] = {
            "model": model,
            "accuracy": acc,
            "predictions": predictions,
            "report": report,
        }

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_name = name

    print(f"\n>>> Best Model: {best_name} (Accuracy: {best_accuracy:.4f})")
    return best_model, best_name, results


def save_plots(y_test, predictions, best_name, df):
    """Generate and save evaluation plots."""
    os.makedirs("static/plots", exist_ok=True)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Low Quality", "High Quality"],
        yticklabels=["Low Quality", "High Quality"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_name}")
    fig.tight_layout()
    fig.savefig("static/plots/confusion_matrix.png", dpi=120)
    plt.close(fig)

    # 2. Label Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["label"].value_counts().sort_index()
    bars = ax.bar(
        ["Low Quality (0)", "High Quality (1)"],
        counts.values,
        color=["#ef4444", "#22c55e"],
    )
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_title("Dataset Label Distribution")
    fig.tight_layout()
    fig.savefig("static/plots/label_distribution.png", dpi=120)
    plt.close(fig)

    # 3. Prompt Length Distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    df["word_count"] = df["prompt"].apply(lambda x: len(str(x).split()))
    low_q = df[df["label"] == 0]["word_count"]
    high_q = df[df["label"] == 1]["word_count"]
    ax.hist(low_q, bins=30, alpha=0.6, label="Low Quality", color="#ef4444")
    ax.hist(high_q, bins=30, alpha=0.6, label="High Quality", color="#22c55e")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    ax.set_title("Prompt Length Distribution by Quality")
    ax.legend()
    fig.tight_layout()
    fig.savefig("static/plots/length_distribution.png", dpi=120)
    plt.close(fig)

    print("\nPlots saved to static/plots/")


def save_model(model, vectorizer):
    """Save trained model and vectorizer."""
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/prompt_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    print("Model saved to models/prompt_model.pkl")
    print("Vectorizer saved to models/tfidf_vectorizer.pkl")


def main():
    print("=" * 60)
    print("  PROMPT QUALITY MODEL TRAINING")
    print("=" * 60)

    # 1. Load data
    df = load_data()
    explore_data(df)

    # 2. Build features
    print("\nBuilding TF-IDF features...")
    X, y, vectorizer = build_features(df)
    print(f"Feature matrix shape: {X.shape}")

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

    # 4. Train and compare models
    best_model, best_name, results = train_and_compare(
        X_train, X_test, y_train, y_test
    )

    # 5. Detailed report for best model
    best_preds = results[best_name]["predictions"]
    print(f"\n--- Detailed Report: {best_name} ---")
    print(classification_report(y_test, best_preds, target_names=["Low Quality", "High Quality"]))

    # 6. Cross-validation
    print("Running 5-fold cross-validation...")
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="accuracy")
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # 7. Save plots
    save_plots(y_test, best_preds, best_name, df)

    # 8. Save model
    save_model(best_model, vectorizer)

    # 9. Quick test
    print("\n--- Quick Prediction Test ---")
    test_prompts = [
        "Explain AI",
        "Tell me about machine learning",
        "Explain artificial intelligence in simple terms. Provide 3 real-world examples. Use bullet points. Limit to 200 words.",
        "You are an expert teacher. Explain deep learning step-by-step with code examples and a summary.",
    ]
    for p in test_prompts:
        vec = vectorizer.transform([p])
        pred = best_model.predict(vec)[0]
        label = "HIGH QUALITY" if pred == 1 else "LOW QUALITY"
        print(f"  [{label:>12}] {p[:70]}...")

    print("\n" + "=" * 60)
    print("  Training complete! Run 'python app.py' to start the API.")
    print("=" * 60)


if __name__ == "__main__":
    main()
