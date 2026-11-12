#!/usr/bin/env python3
"""
Train on the Big Kaggle WELFake Dataset (72,134 articles)
Merges 4 major benchmarks: Kaggle Fake News, McIntire, Reuters, BuzzFeed.
Runs on Apple Silicon MPS or CUDA.
"""

import os
import sys

# Auto-reexec with local .venv if needed
try:
    import torch
    import transformers
    import datasets
    import sklearn
except ImportError:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    venv_py = os.path.join(base_dir, ".venv", "bin", "python")
    scratch_venv_py = "/Users/augnikganguly/.gemini/antigravity/scratch/fake_news_nlp/.venv/bin/python"
    target_py = venv_py if os.path.exists(venv_py) else scratch_venv_py
    if target_py and sys.executable != target_py:
        os.execv(target_py, [target_py] + sys.argv)
    else:
        sys.exit(1)

import time
import json
import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

os.environ['HF_HUB_OFFLINE'] = '1'

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=192):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(int(self.labels[idx]), dtype=torch.long)
        }

def main():
    print("==================================================================")
    print("   TRAINING ON BIG KAGGLE WELFAKE DATASET (72,134 ARTICLES)       ")
    print("==================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Hardware Acceleration: {device.type.upper()}")
    
    # 1. Load WELFake (Word Embedding-based Labeled Fake news Dataset)
    print("\nLoading WELFake dataset...")
    ds = load_dataset('davanstrien/WELFake')
    df = pd.DataFrame(ds['train'])
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    df['full_text'] = df['title'] + ' -- ' + df['text']
    df = df[df['full_text'].str.strip() != ''].reset_index(drop=True)
    
    print(f"Loaded {len(df):,} valid news articles from Kaggle WELFake.")
    print(f"Class Distribution: Real={sum(df['label']==1):,}, Fake={sum(df['label']==0):,}")
    
    # 2. Stratified Train / Test Split
    # Split 80% train / 20% test
    train_df, test_df = train_test_split(df, test_size=0.20, stratify=df['label'], random_state=42)
    print(f"Partitioned into: Train={len(train_df):,} | Test={len(test_df):,}")
    
    # 3. Train Big-Scale Baseline (Sublinear TF-IDF + Logistic Regression)
    print("\n[Step 1/2] Training TF-IDF + Logistic Regression on all 57k+ training articles...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
    X_train = vectorizer.fit_transform(train_df['full_text'])
    X_test = vectorizer.transform(test_df['full_text'])
    
    clf = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
    clf.fit(X_train, train_df['label'].values)
    
    lr_preds = clf.predict(X_test)
    lr_probs = clf.predict_proba(X_test)[:, 1]
    lr_acc = accuracy_score(test_df['label'].values, lr_preds)
    lr_f1 = f1_score(test_df['label'].values, lr_preds, average='macro')
    lr_auc = roc_auc_score(test_df['label'].values, lr_probs)
    lr_time = time.time() - t0
    
    print(f"Logistic Regression Complete ({lr_time:.1f}s):")
    print(f"  Accuracy: {lr_acc*100:.2f}% | Macro F1: {lr_f1:.4f} | ROC-AUC: {lr_auc:.4f}")
    
    # 4. Deep Learning: DistilBERT Training
    # To run smoothly on local Mac, let's take a sample of 12,000 articles for fine-tuning
    print("\n[Step 2/2] Fine-Tuning DistilBERT...")
    db_train_sample = train_df.sample(n=10000, random_state=42)
    db_test_sample = test_df.sample(n=2000, random_state=42)
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.to(device)
    
    train_loader = DataLoader(NewsDataset(db_train_sample['full_text'], db_train_sample['label'], tokenizer, max_length=192), batch_size=16, shuffle=True)
    test_loader = DataLoader(NewsDataset(db_test_sample['full_text'], db_test_sample['label'], tokenizer, max_length=192), batch_size=16, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    print("Fine-tuning 1 Epoch on 10,000 WELFake articles...")
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc="Training DistilBERT"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
        
    print(f"Average Loss: {running_loss / len(train_loader):.4f}")
    
    # Evaluate
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = np.argmax(logits.cpu().numpy(), axis=1)
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(batch['labels'].numpy())
            
    db_acc = accuracy_score(all_labels, all_preds)
    db_f1 = f1_score(all_labels, all_preds, average='macro')
    db_auc = roc_auc_score(all_labels, all_probs)
    
    print("\n" + "=" * 60)
    print("             BIG KAGGLE WELFAKE RESULTS (72K)            ")
    print("=" * 60)
    print(f"Logistic Regression (all 57k train / 14k test) : {lr_acc*100:.2f}% Acc | {lr_f1:.4f} F1 | {lr_auc:.4f} AUC")
    print(f"DistilBERT (Fine-Tuned on WELFake subset)      : {db_acc*100:.2f}% Acc | {db_f1:.4f} F1 | {db_auc:.4f} AUC")
    print("=" * 60)

if __name__ == "__main__":
    main()
