import os
import re
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

os.environ['HF_HUB_OFFLINE'] = '1'

def clean_text(text: str) -> str:
    """Basic text sanitization: normalizes whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def prepare_data(
    train_size: int = 5000,
    val_size: int = 1000,
    test_size: int = 2000,
    seed: int = 42
):
    """
    Loads GonzaloA/fake_news from cache, combines title and text,
    and returns stratified splits.
    
    Labels:
      0: Fake news
      1: Real news
    """
    print(f"Loading dataset from local cache...")
    raw_dataset = load_dataset('GonzaloA/fake_news')
    
    # Extract training and test splits
    train_df = pd.DataFrame(raw_dataset['train'])
    test_df = pd.DataFrame(raw_dataset['test'])
    val_df = pd.DataFrame(raw_dataset['validation'])
    
    # Combine full pool
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df['title'] = full_df['title'].fillna('').apply(clean_text)
    full_df['text'] = full_df['text'].fillna('').apply(clean_text)
    
    # Combined representation: Title + lead content
    full_df['full_text'] = full_df['title'] + " -- " + full_df['text']
    
    # Drop empty records
    full_df = full_df[full_df['full_text'].str.strip() != ''].copy()
    
    # Balance classes in sample
    # Desired total = train_size + val_size + test_size
    total_needed = train_size + val_size + test_size
    n_per_class = total_needed // 2
    
    df_fake = full_df[full_df['label'] == 0].sample(n=n_per_class, random_state=seed)
    df_real = full_df[full_df['label'] == 1].sample(n=n_per_class, random_state=seed)
    
    sampled_df = pd.concat([df_fake, df_real]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    # Stratified train, val, test split
    train_val_df, test_split = train_test_split(
        sampled_df,
        test_size=test_size,
        stratify=sampled_df['label'],
        random_state=seed
    )
    
    train_split, val_split = train_test_split(
        train_val_df,
        test_size=val_size,
        stratify=train_val_df['label'],
        random_state=seed
    )
    
    print(f"Dataset prepared successfully:")
    print(f"  Train: {len(train_split)} samples (Fake: {(train_split['label']==0).sum()}, Real: {(train_split['label']==1).sum()})")
    print(f"  Val:   {len(val_split)} samples (Fake: {(val_split['label']==0).sum()}, Real: {(val_split['label']==1).sum()})")
    print(f"  Test:  {len(test_split)} samples (Fake: {(test_split['label']==0).sum()}, Real: {(test_split['label']==1).sum()})")
    
    return train_split, val_split, test_split

if __name__ == "__main__":
    train_df, val_df, test_df = prepare_data()
    print("Sample 0 full_text[:200]:", train_df.iloc[0]['full_text'][:200])
