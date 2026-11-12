# Veritas • The Style vs. Truth NLP Lab

An investigative natural language processing experiment exploring a simple but fundamental question: **Can sequence classification models actually evaluate factual reality, or do they merely grade how formal an article sounds?**

This repository contains a full pair-wise benchmark: a 66-million parameter fine-tuned **DistilBERT** transformer evaluated side-by-side against a classic **TF-IDF Logistic Regression** baseline on 40,000+ labeled news articles (including the 72,134-article Kaggle WELFake dataset).

---

## ⚡ The 30-Second Quickstart

### Launch the Web Interface
To open the interactive research lab in your browser:
```bash
cd ~/Desktop/fake_news_classification_project
python3 app.py
```
*(Your default web browser will automatically open `http://127.0.0.1:5000`)*

### Test a Headline Directly from Terminal
```bash
python3 demo_predict.py "WASHINGTON (Reuters) - NASA confirmed that the Moon is made of cheese on Wednesday."
```

### Open in VS Code
```bash
code .
```
*(Hit **F5** in VS Code to launch the web studio or any experiment script directly).*

---

## 🔍 The Core Discovery: The "Reuters Trap"

Both models perform with stellar in-distribution scores on benchmark datasets:
- **DistilBERT**: **98.75% Test Accuracy**, **0.9995 ROC-AUC** (25 errors out of 2,000)
- **Logistic Regression**: **96.85% Test Accuracy**, **0.9913 ROC-AUC** (63 errors out of 2,000)

However, inspecting the learned weights revealed why:
1. **The Single-Source Bias**: In the benchmark corpus, **97.2% of verified real articles were scraped directly from Reuters wire feeds**.
2. **The Shortcut**: The model didn't learn what was true; it learned to look for wire datelines (`"WASHINGTON (Reuters) -"`, day-of-week markers like `"wednesday"`, and attribution tags like `"told reuters"`).
3. **The Inverse Reality Trap**:
   - Fabricate a total absurdity wrapped in Reuters style (*"The IMF announced the Moon is made of concentrated dairy cheese"*), and DistilBERT predicts **REAL with 99.9% confidence**.
   - Take a 100% verified historical event (*the 2013 Russian Chelyabinsk meteor explosion that injured over 1,000 people*) and write it in breathless, all-caps clickbait style, and the model rejects it as **FAKE with 99.7% confidence**.

---

## 🧠 Why Sequence Classification Cannot Verify Truth

A text classification model computes the statistical conditional probability:
$$P(Y = 1 \mid X = \text{tokens})$$

This measures **stylistic alignment with the training corpus**—nothing more.

In formal epistemology, truth is an extrinsic property requiring correspondence with the external world:
$$\text{True}(S) \iff \text{World} \models S$$

Because text classification operates entirely in a closed token space without access to live reality, sensor data, or external knowledge bases:
- It gives polished state-sponsored propaganda a free pass.
- It penalizes emotional eyewitness accounts and grassroots whistleblowers.
- It cannot detect temporal drift (a statement like *"Rishi Sunak is the UK Prime Minister"* changes truth value over time without surface syntax changes).

---

## 📂 Repository Structure

```
fake_news_classification_project/
├── app.py                              # Veritas Web Dashboard (Flask backend)
├── templates/index.html                # Editorial, bespoke Web Studio interface
├── README.md                           # Project documentation & findings
├── REPORT.md                           # Formal research paper with math & case studies
├── requirements.txt                    # Minimal Python dependencies
├── fake_news_classification.ipynb      # Step-by-step interactive Jupyter Notebook
├── kaggle_welfake_distilbert.ipynb     # Kaggle GPU (T4/P100) training notebook
├── demo_predict.py                     # CLI tool for interactive text testing
├── data_loader.py                      # Dataset preparation & stratified splitting
├── train_baseline.py                   # Sublinear TF-IDF + Logistic Regression
├── train_distilbert.py                 # DistilBERT fine-tuning on Apple Silicon (MPS)
├── train_welfake_big.py                # 72k Kaggle WELFake dataset trainer
├── adversarial_probes.py               # 5 hypothesis-driven style-vs-truth probes
├── error_analysis.py                   # False positive/negative comparative analysis
├── generate_plots.py                   # High-resolution chart generator
├── run_pipeline.py                     # End-to-end automated pipeline
├── .vscode/                            # One-click VS Code settings & debug launch profiles
├── models/                             # Saved checkpoints (DistilBERT + LR weights)
└── results/                            # Metrics JSONs and visualization charts
```

---

## 🛠️ Requirements & Reproducibility
Runs locally on macOS (with Apple Silicon Metal acceleration) or Linux/Windows with CUDA.
```bash
source .venv/bin/activate
pip install -r requirements.txt
```
