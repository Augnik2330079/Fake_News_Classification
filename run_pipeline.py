import os
import sys

# Auto-reexec with local .venv if dependencies are missing from active environment
try:
    import torch
    import transformers
    import datasets
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
        print("\n[Error] Required dependencies (torch, transformers, datasets, sklearn) are missing.")
        print("Please activate the project virtual environment before running:")
        print("    source .venv/bin/activate")
        print("    python run_pipeline.py\n")
        sys.exit(1)

import json
import time
from data_loader import prepare_data
from train_baseline import train_and_eval_logistic_regression
from train_distilbert import train_and_eval_distilbert
from adversarial_probes import run_probes
from error_analysis import perform_error_analysis

def main():
    print("==================================================================")
    print("    FAKE NEWS CLASSIFICATION & EPISTEMIC ANALYSIS PIPELINE       ")
    print("==================================================================")
    
    start_time = time.time()
    results_dir = "./results"
    models_dir = "./models"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Prepare Data
    train_df, val_df, test_df = prepare_data(
        train_size=5000,
        val_size=1000,
        test_size=2000,
        seed=42
    )
    
    # 2. Train Logistic Regression Baseline
    lr_start = time.time()
    lr_clf, tfidf_vec, lr_results = train_and_eval_logistic_regression(
        train_df, val_df, test_df, output_dir=models_dir
    )
    lr_duration = time.time() - lr_start
    lr_results['training_time_sec'] = lr_duration
    
    # 3. Fine-Tune DistilBERT
    db_start = time.time()
    db_model, tokenizer, db_results = train_and_eval_distilbert(
        train_df, val_df, test_df,
        model_name="distilbert-base-uncased",
        output_dir=os.path.join(models_dir, "distilbert"),
        epochs=2,
        batch_size=16,
        lr=2e-5,
        max_length=256
    )
    db_duration = time.time() - db_start
    db_results['training_time_sec'] = db_duration
    
    # 4. Run Adversarial Probes (Style vs. Factual Ground Truth)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    probe_results = run_probes(lr_clf, tfidf_vec, db_model, tokenizer, device)
    
    # 5. In-Depth Comparative Error Analysis
    style_summary, fp_samples, fn_samples, both_failed = perform_error_analysis(
        test_df,
        lr_results['test_preds'],
        lr_results['test_probs'],
        db_results['test_preds'],
        db_results['test_probs'],
        output_dir=results_dir
    )
    
    # 6. Consolidate and Save Full Results
    total_pipeline_time = time.time() - start_time
    consolidated_results = {
        "pipeline_execution_time_sec": total_pipeline_time,
        "dataset_split": {
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df)
        },
        "logistic_regression": {
            "accuracy": lr_results["accuracy"],
            "macro_f1": lr_results["macro_f1"],
            "roc_auc": lr_results["roc_auc"],
            "fake_precision": lr_results["fake_precision"],
            "fake_recall": lr_results["fake_recall"],
            "fake_f1": lr_results["fake_f1"],
            "real_precision": lr_results["real_precision"],
            "real_recall": lr_results["real_recall"],
            "real_f1": lr_results["real_f1"],
            "confusion_matrix": lr_results["confusion_matrix"],
            "training_time_sec": lr_results["training_time_sec"],
            "top_real_features": lr_results["top_real_features"],
            "top_fake_features": lr_results["top_fake_features"]
        },
        "distilbert": {
            "accuracy": db_results["accuracy"],
            "macro_f1": db_results["macro_f1"],
            "roc_auc": db_results["roc_auc"],
            "fake_precision": db_results["fake_precision"],
            "fake_recall": db_results["fake_recall"],
            "fake_f1": db_results["fake_f1"],
            "real_precision": db_results["real_precision"],
            "real_recall": db_results["real_recall"],
            "real_f1": db_results["real_f1"],
            "confusion_matrix": db_results["confusion_matrix"],
            "training_time_sec": db_results["training_time_sec"],
            "history": db_results["history"]
        },
        "adversarial_probes": probe_results,
        "style_summary": style_summary
    }
    
    with open(os.path.join(results_dir, "consolidated_metrics.json"), "w") as f:
        json.dump(consolidated_results, f, indent=2)
        
    print("\n==================================================================")
    print("                    PIPELINE COMPLETED!                          ")
    print("==================================================================")
    print(f"Total time elapsed: {total_pipeline_time:.2f} seconds")
    print(f"Metrics saved to:   {os.path.join(results_dir, 'consolidated_metrics.json')}")
    print(f"Predictions saved:  {os.path.join(results_dir, 'test_predictions_detailed.csv')}")

if __name__ == "__main__":
    main()
