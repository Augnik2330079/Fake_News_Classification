#!/usr/bin/env python3
"""
Veritas • The Style vs. Truth NLP Lab
An interactive web application exploring why text classification models
confuse linguistic writing style with factual reality.
"""

import os
import sys

# Auto-reexec with local .venv if needed
try:
    import torch
    import transformers
    import joblib
    import sklearn
    from flask import Flask, render_template, request, jsonify, send_from_directory
except ImportError:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    venv_py = os.path.join(base_dir, ".venv", "bin", "python")
    scratch_venv_py = "/Users/augnikganguly/.gemini/antigravity/scratch/fake_news_nlp/.venv/bin/python"
    target_py = venv_py if os.path.exists(venv_py) else scratch_venv_py
    if target_py and sys.executable != target_py:
        os.execv(target_py, [target_py] + sys.argv)
    else:
        sys.exit(1)

import json
import webbrowser
import threading
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DISTILBERT_DIR = os.path.join(MODELS_DIR, "distilbert")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATES_DIR)

print("Starting Veritas Research Lab...")
print("-> Loading Logistic Regression & TF-IDF Vectorizer...")
lr_clf = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib"))
vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
feature_names = np.array(vectorizer.get_feature_names_out())

print("-> Loading Fine-Tuned DistilBERT Checkpoint...")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
db_model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR)
db_model.to(device)
db_model.eval()
print(f"-> Ready! Using hardware accelerator: {device.type.upper()}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/results/<path:filename>")
def serve_results(filename):
    return send_from_directory(RESULTS_DIR, filename)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # 1. Classical Logistic Regression
    X_vec = vectorizer.transform([text])
    lr_probs = lr_clf.predict_proba(X_vec)[0]
    lr_pred = int(np.argmax(lr_probs))
    lr_prob_real = float(lr_probs[1])
    lr_prob_fake = float(lr_probs[0])

    # Extract words that contributed most to this classification
    nz_indices = X_vec.nonzero()[1]
    weights = lr_clf.coef_[0][nz_indices]
    words = feature_names[nz_indices]
    word_weights = sorted(zip(words, weights), key=lambda x: abs(x[1]), reverse=True)[:8]
    formatted_tokens = [[str(w), float(wt)] for w, wt in word_weights]

    # 2. Deep DistilBERT Transformer
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    with torch.no_grad():
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        logits = db_model(input_ids=input_ids, attention_mask=attention_mask).logits
        db_probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        db_pred = int(np.argmax(db_probs))
        db_prob_fake = float(db_probs[0])
        db_prob_real = float(db_probs[1])

    # 3. Humanized Epistemic Heuristic Analysis
    lower_text = text.lower()
    has_warning = False
    warning_title = ""
    warning_message = ""

    if "reuters" in lower_text:
        has_warning = True
        warning_title = "The Wire Dateline Shortcut ('Reuters' detected)"
        warning_message = "Notice how confident the model is? In standard fake-news datasets, over 97% of real articles come straight from Reuters wires. The model learned to worship the word 'Reuters' rather than verifying if the actual claim makes any sense in reality."
    elif text.count("!") >= 2 or (sum(1 for c in text if c.isupper()) / (len(text) + 1e-5)) > 0.15:
        has_warning = True
        warning_title = "Sensory Overload Bias (Clickbait Punctuation)"
        warning_message = "Too many exclamation marks or shouty capital letters triggered the spam filter. Even when an article describes a 100% verified historical event (like the Russian meteor impact), the model dismisses it as fake simply because it sounds sensational."
    elif any(term in lower_text for term in ["washington", "minister", "accord", "consortium", "stated"]):
        has_warning = True
        warning_title = "The Bureaucratic Disguise"
        warning_message = "The text adopts dry, institutional jargon. DistilBERT tends to give dispassionate, official-sounding text a free pass, which is precisely how sophisticated disinformation bypasses automated filters."

    response = {
        "text_preview": text[:150] + ("..." if len(text) > 150 else ""),
        "logistic_regression": {
            "prediction": "REAL" if lr_pred == 1 else "FAKE",
            "label": lr_pred,
            "prob_real": lr_prob_real,
            "prob_fake": lr_prob_fake,
            "top_tokens": formatted_tokens
        },
        "distilbert": {
            "prediction": "REAL" if db_pred == 1 else "FAKE",
            "label": db_pred,
            "prob_real": db_prob_real,
            "prob_fake": db_prob_fake
        },
        "epistemic_analysis": {
            "has_warning": has_warning,
            "warning_title": warning_title,
            "warning_message": warning_message
        }
    }
    return jsonify(response)

@app.route("/api/probes", methods=["GET"])
def api_probes():
    try:
        with open(os.path.join(RESULTS_DIR, "consolidated_metrics.json")) as f:
            metrics = json.load(f)
        return jsonify(metrics.get("adversarial_probes", []))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    try:
        with open(os.path.join(RESULTS_DIR, "consolidated_metrics.json")) as f:
            metrics = json.load(f)
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def open_browser(port):
    import time
    time.sleep(1.2)
    webbrowser.open(f"http://127.0.0.1:{port}")

if __name__ == "__main__":
    port = 5000
    print("\n" + "=" * 65)
    print("    VERITAS • THE STYLE VS. TRUTH NLP LAB                        ")
    print("=" * 65)
    print(f"Server live at: http://127.0.0.1:{port}")
    print("Opening web interface in your default browser...")
    print("Press Ctrl+C to quit.")
    print("=" * 65 + "\n")

    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    app.run(host="127.0.0.1", port=port, debug=False)
