import os
import json
import pandas as pd
import numpy as np

def perform_error_analysis(test_df, lr_preds, lr_probs, db_preds, db_probs, output_dir="./results"):
    os.makedirs(output_dir, exist_ok=True)
    
    analysis_df = test_df.copy().reset_index(drop=True)
    analysis_df['lr_pred'] = lr_preds
    analysis_df['lr_prob_real'] = lr_probs
    analysis_df['db_pred'] = db_preds
    analysis_df['db_prob_real'] = db_probs
    
    # Define error flags
    # Label 0: Fake, Label 1: Real
    analysis_df['lr_correct'] = (analysis_df['label'] == analysis_df['lr_pred'])
    analysis_df['db_correct'] = (analysis_df['label'] == analysis_df['db_pred'])
    
    # False Positives: Fake News (0) incorrectly predicted as Real News (1)
    # (High risk: allowing disinformation through as credible)
    lr_fp = analysis_df[(analysis_df['label'] == 0) & (analysis_df['lr_pred'] == 1)]
    db_fp = analysis_df[(analysis_df['label'] == 0) & (analysis_df['db_pred'] == 1)]
    
    # False Negatives: Real News (1) incorrectly predicted as Fake News (0)
    # (High risk: censoring legitimate journalism)
    lr_fn = analysis_df[(analysis_df['label'] == 1) & (analysis_df['lr_pred'] == 0)]
    db_fn = analysis_df[(analysis_df['label'] == 1) & (analysis_df['db_pred'] == 0)]
    
    # DistilBERT wins over LR
    db_wins = analysis_df[(~analysis_df['lr_correct']) & (analysis_df['db_correct'])]
    
    # LR wins over DistilBERT
    lr_wins = analysis_df[(analysis_df['lr_correct']) & (~analysis_df['db_correct'])]
    
    # Both failed
    both_failed = analysis_df[(~analysis_df['lr_correct']) & (~analysis_df['db_correct'])]
    
    print("\n========================================================")
    print("                DETAILED ERROR ANALYSIS                 ")
    print("========================================================")
    print(f"Total Test Set Samples: {len(analysis_df)}")
    print(f"Logistic Regression Errors : {(~analysis_df['lr_correct']).sum()} (FP: {len(lr_fp)}, FN: {len(lr_fn)})")
    print(f"DistilBERT Errors          : {(~analysis_df['db_correct']).sum()} (FP: {len(db_fp)}, FN: {len(db_fn)})")
    print(f"DistilBERT Corrected LR    : {len(db_wins)} cases")
    print(f"LR Corrected DistilBERT    : {len(lr_wins)} cases")
    print(f"Both Models Failed         : {len(both_failed)} cases")
    
    # Qualitative analysis of DistilBERT False Positives (Fake predicted as Real)
    print("\n--- DistilBERT False Positives (Fake News Predicted as Real) ---")
    fp_samples = []
    for idx, row in db_fp.head(5).iterrows():
        title = row['title'][:80]
        text_preview = row['text'][:150]
        prob = row['db_prob_real']
        print(f" [FP #{idx}] P(Real)={prob:.4f} | Title: {title}")
        print(f"   Excerpt: {text_preview}...")
        fp_samples.append({
            'index': int(idx),
            'title': row['title'],
            'text_preview': text_preview,
            'label': int(row['label']),
            'db_prob_real': float(prob),
            'lr_prob_real': float(row['lr_prob_real'])
        })
        
    # Qualitative analysis of DistilBERT False Negatives (Real News Predicted as Fake)
    print("\n--- DistilBERT False Negatives (Real News Predicted as Fake) ---")
    fn_samples = []
    for idx, row in db_fn.head(5).iterrows():
        title = row['title'][:80]
        text_preview = row['text'][:150]
        prob = row['db_prob_real']
        print(f" [FN #{idx}] P(Real)={prob:.4f} | Title: {title}")
        print(f"   Excerpt: {text_preview}...")
        fn_samples.append({
            'index': int(idx),
            'title': row['title'],
            'text_preview': text_preview,
            'label': int(row['label']),
            'db_prob_real': float(prob),
            'lr_prob_real': float(row['lr_prob_real'])
        })
        
    # Linguistic and stylistic metrics across error types
    def get_style_stats(df_subset):
        if len(df_subset) == 0:
            return {}
        text_lens = df_subset['full_text'].apply(lambda x: len(x.split()))
        caps_ratio = df_subset['full_text'].apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1e-6))
        exclamations = df_subset['full_text'].apply(lambda x: x.count('!'))
        reuters_mention = df_subset['full_text'].apply(lambda x: 1 if 'reuters' in x.lower() else 0)
        return {
            'count': int(len(df_subset)),
            'avg_word_count': float(text_lens.mean()),
            'avg_caps_ratio': float(caps_ratio.mean()),
            'avg_exclamations': float(exclamations.mean()),
            'reuters_mention_pct': float(reuters_mention.mean() * 100)
        }
        
    style_summary = {
        'all_test': get_style_stats(analysis_df),
        'real_class': get_style_stats(analysis_df[analysis_df['label'] == 1]),
        'fake_class': get_style_stats(analysis_df[analysis_df['label'] == 0]),
        'distilbert_fp': get_style_stats(db_fp),
        'distilbert_fn': get_style_stats(db_fn),
        'lr_fp': get_style_stats(lr_fp),
        'lr_fn': get_style_stats(lr_fn),
        'both_failed': get_style_stats(both_failed)
    }
    
    # Save error analysis output
    with open(os.path.join(output_dir, "error_analysis.json"), "w") as f:
        json.dump({
            'summary': {
                'total_test': len(analysis_df),
                'lr_errors': int((~analysis_df['lr_correct']).sum()),
                'db_errors': int((~analysis_df['db_correct']).sum()),
                'lr_fp': len(lr_fp),
                'lr_fn': len(lr_fn),
                'db_fp': len(db_fp),
                'db_fn': len(db_fn),
                'db_wins': len(db_wins),
                'lr_wins': len(lr_wins),
                'both_failed': len(both_failed)
            },
            'style_summary': style_summary,
            'db_fp_samples': fp_samples,
            'db_fn_samples': fn_samples
        }, f, indent=2)
        
    analysis_df.to_csv(os.path.join(output_dir, "test_predictions_detailed.csv"), index=False)
    print(f"\nSaved detailed predictions to {os.path.join(output_dir, 'test_predictions_detailed.csv')}")
    print(f"Saved error analysis summary to {os.path.join(output_dir, 'error_analysis.json')}")
    
    return style_summary, fp_samples, fn_samples, both_failed
