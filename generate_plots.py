import os
import json
import matplotlib.pyplot as plt
import numpy as np

def generate_visualizations(results_path="./results/consolidated_metrics.json", output_dir="./results"):
    os.makedirs(output_dir, exist_ok=True)
    with open(results_path, "r") as f:
        metrics = json.load(f)
        
    # Plot 1: Confusion Matrices Comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    cm_lr = np.array(metrics['logistic_regression']['confusion_matrix'])
    cm_db = np.array(metrics['distilbert']['confusion_matrix'])
    
    for ax, cm, title, acc in zip(
        axes,
        [cm_lr, cm_db],
        ['Logistic Regression (TF-IDF)', 'DistilBERT (Fine-Tuned)'],
        [metrics['logistic_regression']['accuracy'], metrics['distilbert']['accuracy']]
    ):
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f"{title}\nAccuracy: {acc*100:.2f}%", fontsize=12, fontweight='bold', pad=10)
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(['Fake (0)', 'Real (1)'], fontsize=10)
        ax.set_yticklabels(['Fake (0)', 'Real (1)'], fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)
        
        # Numbers
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=12, fontweight='bold')
                        
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    
    # Plot 2: Top Features for Logistic Regression
    top_real = metrics['logistic_regression']['top_real_features'][:10]
    top_fake = metrics['logistic_regression']['top_fake_features'][:10]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    words = [w[0] for w in top_fake[::-1]] + [w[0] for w in top_real]
    scores = [w[1] for w in top_fake[::-1]] + [w[1] for w in top_real]
    colors = ['#e74c3c' if s < 0 else '#2ecc71' for s in scores]
    
    y_pos = np.arange(len(words))
    ax.barh(y_pos, scores, color=colors, edgecolor='none', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words, fontsize=10, fontweight='medium')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Logistic Regression Coefficient (Negative=Fake, Positive=Real)', fontsize=11)
    ax.set_title('Top Indicative N-grams in Fake News Dataset (TF-IDF + LR)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    feat_path = os.path.join(output_dir, "top_features.png")
    plt.savefig(feat_path, dpi=200)
    plt.close()
    
    # Plot 3: Adversarial Probes Confidence Discrepancy
    probes = metrics['adversarial_probes']
    probe_labels = [
        "P1: Moon Cheese\n(Fake in Wire Style)",
        "P2: Meteor Blast\n(Real in Clickbait Style)",
        "P3: 5G Necrosis\n(Fake in Clinical Style)",
        "P4: Postpone Winter\n(Satire in Wire Style)",
        "P5: JWST Align\n(Real in Casual Style)"
    ]
    db_conf_real = [p['distilbert_prob_real'] * 100 for p in probes]
    lr_conf_real = [p['lr_prob_real'] * 100 for p in probes]
    
    x = np.arange(len(probes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(11, 5.5))
    rects1 = ax.bar(x - width/2, lr_conf_real, width, label='Logistic Regression P(Real)', color='#3498db')
    rects2 = ax.bar(x + width/2, db_conf_real, width, label='DistilBERT P(Real)', color='#9b59b6')
    
    ax.axhline(50, color='red', linestyle=':', linewidth=1.5, label='Decision Boundary (50%)')
    ax.set_ylabel('Predicted Probability of REAL News (%)', fontsize=11)
    ax.set_title('Adversarial Probes: How Linguistic Style Inverts Model Truth Perception', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(probe_labels, fontsize=9.5)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', frameon=True)
    
    for r in rects1:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5)
    for r in rects2:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
                    
    plt.tight_layout()
    probe_path = os.path.join(output_dir, "adversarial_probes.png")
    plt.savefig(probe_path, dpi=200)
    plt.close()
    
    print(f"Visualizations generated successfully:\n - {cm_path}\n - {feat_path}\n - {probe_path}")

if __name__ == "__main__":
    generate_visualizations()
