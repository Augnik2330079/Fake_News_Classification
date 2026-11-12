import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from tqdm import tqdm

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item() * len(labels)
            
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, acc, auc, f1, np.array(all_preds), np.array(all_probs), np.array(all_labels)

def train_and_eval_distilbert(
    train_df,
    val_df,
    test_df,
    model_name="distilbert-base-uncased",
    output_dir="./models/distilbert",
    epochs=2,
    batch_size=16,
    lr=2e-5,
    max_length=256
):
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n--- Fine-Tuning DistilBERT on {device.type.upper()} ---")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    
    train_dataset = NewsDataset(train_df['full_text'], train_df['label'], tokenizer, max_length=max_length)
    val_dataset = NewsDataset(val_df['full_text'], val_df['label'], tokenizer, max_length=max_length)
    test_dataset = NewsDataset(test_df['full_text'], test_df['label'], tokenizer, max_length=max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    best_val_acc = 0.0
    history = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'train_loss': f"{running_loss / (pbar.n + 1):.4f}"})
            
        epoch_loss = running_loss / len(train_loader)
        
        # Validation
        val_loss, val_acc, val_auc, val_f1, _, _, _ = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch} Results | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")
        
        history.append({
            'epoch': epoch,
            'train_loss': epoch_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_auc': val_auc,
            'val_f1': val_f1
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"--> New best validation accuracy: {best_val_acc:.4f}. Saving checkpoint...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
    print("\n--- Evaluating Best DistilBERT on Held-Out Test Set ---")
    best_model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    best_model.to(device)
    
    test_loss, test_acc, test_auc, test_macro_f1, test_preds, test_probs, test_labels = evaluate_model(
        best_model, test_loader, device
    )
    
    prec_fake = precision_score(test_labels, test_preds, pos_label=0)
    rec_fake = recall_score(test_labels, test_preds, pos_label=0)
    f1_fake = f1_score(test_labels, test_preds, pos_label=0)
    
    prec_real = precision_score(test_labels, test_preds, pos_label=1)
    rec_real = recall_score(test_labels, test_preds, pos_label=1)
    f1_real = f1_score(test_labels, test_preds, pos_label=1)
    
    cm = confusion_matrix(test_labels, test_preds)
    
    print(f"Test Accuracy:       {test_acc:.4f}")
    print(f"Test ROC-AUC:        {test_auc:.4f}")
    print(f"Macro F1-Score:      {test_macro_f1:.4f}")
    print(f"Fake (0) - Prec: {prec_fake:.4f}, Rec: {rec_fake:.4f}, F1: {f1_fake:.4f}")
    print(f"Real (1) - Prec: {prec_real:.4f}, Rec: {rec_real:.4f}, F1: {f1_real:.4f}")
    print("Confusion Matrix (rows=actual, cols=pred):")
    print(f"  [TN={cm[0,0]} (Fake as Fake), FP={cm[0,1]} (Fake as Real)]")
    print(f"  [FN={cm[1,0]} (Real as Fake), TP={cm[1,1]} (Real as Real)]")
    
    results = {
        "model": "DistilBERT (Fine-Tuned)",
        "val_accuracy": float(best_val_acc),
        "accuracy": float(test_acc),
        "macro_f1": float(test_macro_f1),
        "roc_auc": float(test_auc),
        "fake_precision": float(prec_fake),
        "fake_recall": float(rec_fake),
        "fake_f1": float(f1_fake),
        "real_precision": float(prec_real),
        "real_recall": float(rec_real),
        "real_f1": float(f1_real),
        "confusion_matrix": cm.tolist(),
        "history": history,
        "test_preds": test_preds.tolist(),
        "test_probs": test_probs.tolist()
    }
    
    return best_model, tokenizer, results

if __name__ == "__main__":
    from data_loader import prepare_data
    train_df, val_df, test_df = prepare_data()
    train_and_eval_distilbert(train_df, val_df, test_df)
