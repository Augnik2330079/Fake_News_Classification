#!/usr/bin/env python3
"""
Interactive Inference & Style-vs-Truth Probe Tool
Tests any custom headline or news article text against both:
  1. TF-IDF + Logistic Regression
  2. Fine-Tuned DistilBERT
"""

import os
import sys

# Auto-reexec with local .venv if dependencies are missing from active environment
try:
    import torch
    import transformers
    import joblib
    import sklearn
except ImportError:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    venv_py = os.path.join(base_dir, ".venv", "bin", "python")
    scratch_venv_py = "/Users/augnikganguly/.gemini/antigravity/scratch/fake_news_nlp/.venv/bin/python"
    
    target_py = None
    if os.path.exists(venv_py):
        target_py = venv_py
    elif os.path.exists(scratch_venv_py):
        target_py = scratch_venv_py
        
    if target_py and sys.executable != target_py:
        os.execv(target_py, [target_py] + sys.argv)
    else:
        print("\n[Error] Required dependencies (torch, transformers, sklearn) are missing.")
        print("Please activate the project virtual environment before running:")
        print("    source .venv/bin/activate")
        print("    python demo_predict.py\n")
        sys.exit(1)

import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DISTILBERT_DIR = os.path.join(MODELS_DIR, "distilbert")

def load_models():
    print("Loading Logistic Regression baseline...")
    lr_clf = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    
    print("Loading fine-tuned DistilBERT model...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
    db_model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR)
    db_model.to(device)
    db_model.eval()
    
    return lr_clf, vectorizer, db_model, tokenizer, device

def predict_article(text: str, lr_clf, vectorizer, db_model, tokenizer, device):
    # 1. Logistic Regression
    X_vec = vectorizer.transform([text])
    lr_probs = lr_clf.predict_proba(X_vec)[0]
    lr_pred = int(np.argmax(lr_probs))
    lr_prob_real = float(lr_probs[1])
    lr_prob_fake = float(lr_probs[0])
    
    # Extract top activated features for this text in LR
    feature_names = np.array(vectorizer.get_feature_names_out())
    nz_indices = X_vec.nonzero()[1]
    weights = lr_clf.coef_[0][nz_indices]
    words = feature_names[nz_indices]
    word_weights = sorted(zip(words, weights), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    # 2. DistilBERT
    enc = tokenizer(text, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
    with torch.no_grad():
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        logits = db_model(input_ids=input_ids, attention_mask=attention_mask).logits
        db_probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        db_pred = int(np.argmax(db_probs))
        db_prob_fake = float(db_probs[0])
        db_prob_real = float(db_probs[1])
        
    print("\n" + "=" * 65)
    print("                     PREDICTION RESULTS                          ")
    print("=" * 65)
    print(f"Input Preview: \"{text[:120]}...\"\n")
    
    print("[1] TF-IDF + Logistic Regression:")
    print(f"    - Prediction : {'REAL NEWS (1)' if lr_pred == 1 else 'FAKE NEWS (0)'}")
    print(f"    - Confidence : {max(lr_prob_real, lr_prob_fake)*100:.2f}%")
    print(f"    - P(Real)    : {lr_prob_real*100:.2f}% | P(Fake): {lr_prob_fake*100:.2f}%")
    print(f"    - Top Influential Tokens : {', '.join([f'{w} ({val:+.2f})' for w, val in word_weights])}")
    
    print("\n[2] Fine-Tuned DistilBERT:")
    print(f"    - Prediction : {'REAL NEWS (1)' if db_pred == 1 else 'FAKE NEWS (0)'}")
    print(f"    - Confidence : {max(db_prob_real, db_prob_fake)*100:.2f}%")
    print(f"    - P(Real)    : {db_prob_real*100:.2f}% | P(Fake): {db_prob_fake*100:.2f}%")
    
    print("\n[!] Epistemic Cautionary Note:")
    if 'reuters' in text.lower():
        print("    -> Warning: Input text mentions 'Reuters'. The model heavily weights wire-service")
        print("       markers and may predict 'REAL' even if the claim is completely counterfactual.")
    if '!' in text or text.isupper():
        print("    -> Warning: Input text contains high-intensity punctuation/capitalization.")
        print("       The model may predict 'FAKE' purely due to stylistic tabloid cues.")
    print("    -> Reminder: These models score stylistic alignment with training corpus sources.")
    print("       They DO NOT verify whether the underlying claim is true in the real world.")
    print("=" * 65 + "\n")

def main():
    lr_clf, vectorizer, db_model, tokenizer, device = load_models()
    print("Models loaded successfully. Ready for inference!")
    
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        predict_article(input_text, lr_clf, vectorizer, db_model, tokenizer, device)
        return

    print("\nInteractive Mode: Enter article text or headline to classify (or 'quit' to exit):")
    while True:
        try:
            user_input = input("\nEnter news text > ").strip()
            if not user_input or user_input.lower() in ('quit', 'exit', 'q'):
                print("Exiting demo. Goodbye!")
                break
            predict_article(user_input, lr_clf, vectorizer, db_model, tokenizer, device)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting demo.")
            break

if __name__ == "__main__":
    main()
