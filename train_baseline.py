import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

def train_and_eval_logistic_regression(train_df, val_df, test_df, output_dir="./models"):
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n--- Training Logistic Regression Baseline (TF-IDF + LR) ---")
    
    # 1. Feature Extraction: Sublinear TF-IDF with Unigrams and Bigrams
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True
    )
    
    X_train = vectorizer.fit_transform(train_df['full_text'])
    y_train = train_df['label'].values
    
    X_val = vectorizer.transform(val_df['full_text'])
    y_val = val_df['label'].values
    
    X_test = vectorizer.transform(test_df['full_text'])
    y_test = test_df['label'].values
    
    # 2. Train Logistic Regression
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    # 3. Predict on Test Set
    val_preds = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    
    test_preds = clf.predict(X_test)
    test_probs = clf.predict_proba(X_test)[:, 1]
    
    # 4. Compute Metrics
    acc = accuracy_score(y_test, test_preds)
    prec_fake = precision_score(y_test, test_preds, pos_label=0)
    rec_fake = recall_score(y_test, test_preds, pos_label=0)
    f1_fake = f1_score(y_test, test_preds, pos_label=0)
    
    prec_real = precision_score(y_test, test_preds, pos_label=1)
    rec_real = recall_score(y_test, test_preds, pos_label=1)
    f1_real = f1_score(y_test, test_preds, pos_label=1)
    
    macro_f1 = f1_score(y_test, test_preds, average='macro')
    auc = roc_auc_score(y_test, test_probs)
    cm = confusion_matrix(y_test, test_preds)
    
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy:       {acc:.4f}")
    print(f"Test ROC-AUC:        {auc:.4f}")
    print(f"Macro F1-Score:      {macro_f1:.4f}")
    print(f"Fake (0) - Prec: {prec_fake:.4f}, Rec: {rec_fake:.4f}, F1: {f1_fake:.4f}")
    print(f"Real (1) - Prec: {prec_real:.4f}, Rec: {rec_real:.4f}, F1: {f1_real:.4f}")
    print("Confusion Matrix (rows=actual, cols=pred):")
    print(f"  [TN={cm[0,0]} (Fake as Fake), FP={cm[0,1]} (Fake as Real)]")
    print(f"  [FN={cm[1,0]} (Real as Fake), TP={cm[1,1]} (Real as Real)]")
    
    # 5. Extract top predictive features
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0]
    
    # Top features for Real news (class 1 -> highest positive weights)
    top_real_idx = np.argsort(coefs)[-20:][::-1]
    top_real_features = [(feature_names[i], float(coefs[i])) for i in top_real_idx]
    
    # Top features for Fake news (class 0 -> lowest negative weights)
    top_fake_idx = np.argsort(coefs)[:20]
    top_fake_features = [(feature_names[i], float(coefs[i])) for i in top_fake_idx]
    
    # 6. Save artifacts
    joblib.dump(clf, os.path.join(output_dir, "logistic_regression.joblib"))
    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.joblib"))
    
    results = {
        "model": "TF-IDF + Logistic Regression",
        "val_accuracy": float(val_acc),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "roc_auc": float(auc),
        "fake_precision": float(prec_fake),
        "fake_recall": float(rec_fake),
        "fake_f1": float(f1_fake),
        "real_precision": float(prec_real),
        "real_recall": float(rec_real),
        "real_f1": float(f1_real),
        "confusion_matrix": cm.tolist(),
        "top_real_features": top_real_features,
        "top_fake_features": top_fake_features,
        "test_preds": test_preds.tolist(),
        "test_probs": test_probs.tolist()
    }
    
    return clf, vectorizer, results

if __name__ == "__main__":
    from data_loader import prepare_data
    train_df, val_df, test_df = prepare_data()
    train_and_eval_logistic_regression(train_df, val_df, test_df)
