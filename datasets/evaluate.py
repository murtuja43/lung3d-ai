import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score
)
from datasets.dataloader import get_dataloaders
from datasets.model import TBClassifierWithFeatures, get_device

# ─────────────────────────────────────────
# Load model + run inference on test set
# ─────────────────────────────────────────
def get_predictions(model, loader, device):
    model.eval()
    all_preds   = []
    all_labels  = []
    all_probs   = []

    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(device)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            preds   = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )


# ─────────────────────────────────────────
# Print full classification report
# ─────────────────────────────────────────
def print_metrics(labels, preds, probs):
    print("\n" + "="*60)
    print("📊 FULL EVALUATION REPORT — Lung3D AI")
    print("="*60)

    # Classification report
    report = classification_report(
        labels, preds,
        target_names=['Normal', 'TB'],
        digits=4
    )
    print("\n📋 Classification Report:")
    print(report)

    # Confusion matrix values
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    print("📊 Confusion Matrix Values:")
    print(f"   True Negatives  (TN): {tn}")
    print(f"   False Positives (FP): {fp}")
    print(f"   False Negatives (FN): {fn}")
    print(f"   True Positives  (TP): {tp}")

    # Clinical metrics
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv         = tp / (tp + fp)
    npv         = tn / (tn + fn)
    f1          = 2 * (ppv * sensitivity) / (ppv + sensitivity)

    # ROC AUC
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc     = auc(fpr, tpr)

    print(f"\n🏥 Clinical Metrics:")
    print(f"   Sensitivity (Recall)  : {sensitivity:.4f} ({sensitivity*100:.2f}%)")
    print(f"   Specificity           : {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"   Precision (PPV)       : {ppv:.4f} ({ppv*100:.2f}%)")
    print(f"   NPV                   : {npv:.4f} ({npv*100:.2f}%)")
    print(f"   F1 Score              : {f1:.4f}")
    print(f"   ROC AUC               : {roc_auc:.4f}")
    print("="*60)

    return cm, fpr, tpr, roc_auc


# ─────────────────────────────────────────
# Plot everything in one figure
# ─────────────────────────────────────────
def plot_evaluation(labels, preds, probs, cm,
                    fpr, tpr, roc_auc, save_dir='models'):

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#0a0e1a')

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        hspace=0.4,
        wspace=0.35
    )

    # ── Color scheme ──
    BG       = '#0a0e1a'
    CARD     = '#111827'
    BORDER   = '#1e2d45'
    ACCENT   = '#00d4ff'
    GREEN    = '#00ff88'
    RED      = '#ff4757'
    YELLOW   = '#ffa502'
    TEXT     = '#e8f4fd'
    TEXT_DIM = '#8899aa'

    def style_ax(ax, title):
        ax.set_facecolor(CARD)
        ax.tick_params(colors=TEXT_DIM, labelsize=9)
        ax.xaxis.label.set_color(TEXT_DIM)
        ax.yaxis.label.set_color(TEXT_DIM)
        ax.title.set_color(ACCENT)
        ax.set_title(title, fontsize=12,
                     fontweight='bold', pad=12)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

    # ── 1. Confusion Matrix ──
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, '🔢 Confusion Matrix')

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = ax1.imshow(cm_norm, interpolation='nearest',
                    cmap='Blues', vmin=0, vmax=1)

    classes = ['Normal', 'TB']
    tick_marks = np.arange(len(classes))
    ax1.set_xticks(tick_marks)
    ax1.set_yticks(tick_marks)
    ax1.set_xticklabels(classes, color=TEXT)
    ax1.set_yticklabels(classes, color=TEXT)
    ax1.set_xlabel('Predicted Label', color=TEXT_DIM)
    ax1.set_ylabel('True Label', color=TEXT_DIM)

    for i in range(2):
        for j in range(2):
            val     = cm[i, j]
            pct     = cm_norm[i, j]
            color   = 'white' if pct > 0.5 else TEXT_DIM
            ax1.text(j, i,
                     f'{val}\n({pct*100:.1f}%)',
                     ha='center', va='center',
                     color=color, fontsize=12,
                     fontweight='bold')

    plt.colorbar(im, ax=ax1, fraction=0.046)

    # ── 2. ROC Curve ──
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, '📈 ROC Curve')

    ax2.plot(fpr, tpr, color=ACCENT, lw=2,
             label=f'AUC = {roc_auc:.3f}')
    ax2.plot([0,1], [0,1], color=BORDER,
             lw=1, linestyle='--',
             label='Random classifier')
    ax2.fill_between(fpr, tpr, alpha=0.1, color=ACCENT)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend(facecolor=CARD, labelcolor=TEXT,
               fontsize=9)
    ax2.grid(True, alpha=0.1, color=BORDER)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.02])

    # ── 3. Precision-Recall Curve ──
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3, '🎯 Precision-Recall Curve')

    precision_vals, recall_vals, _ = precision_recall_curve(
        labels, probs
    )
    avg_precision = average_precision_score(labels, probs)

    ax3.plot(recall_vals, precision_vals,
             color=GREEN, lw=2,
             label=f'AP = {avg_precision:.3f}')
    ax3.fill_between(recall_vals, precision_vals,
                     alpha=0.1, color=GREEN)
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.legend(facecolor=CARD, labelcolor=TEXT,
               fontsize=9)
    ax3.grid(True, alpha=0.1, color=BORDER)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1.02])

    # ── 4. Per-class Metrics Bar Chart ──
    ax4 = fig.add_subplot(gs[1, 0])
    style_ax(ax4, '📊 Per-Class Metrics')

    report_dict = classification_report(
        labels, preds,
        target_names=['Normal', 'TB'],
        output_dict=True
    )

    metrics      = ['precision', 'recall', 'f1-score']
    normal_vals  = [report_dict['Normal'][m] for m in metrics]
    tb_vals      = [report_dict['TB'][m]     for m in metrics]
    x            = np.arange(len(metrics))
    width        = 0.35

    bars1 = ax4.bar(x - width/2, normal_vals,
                    width, label='Normal',
                    color=GREEN, alpha=0.8)
    bars2 = ax4.bar(x + width/2, tb_vals,
                    width, label='TB',
                    color=RED, alpha=0.8)

    ax4.set_xticks(x)
    ax4.set_xticklabels(
        ['Precision', 'Recall', 'F1'],
        color=TEXT
    )
    ax4.set_ylim([0, 1.1])
    ax4.legend(facecolor=CARD, labelcolor=TEXT,
               fontsize=9)
    ax4.grid(True, alpha=0.1, axis='y',
             color=BORDER)

    for bar in bars1:
        ax4.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f'{bar.get_height():.2f}',
            ha='center', va='bottom',
            color=TEXT, fontsize=8
        )
    for bar in bars2:
        ax4.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f'{bar.get_height():.2f}',
            ha='center', va='bottom',
            color=TEXT, fontsize=8
        )

    # ── 5. Confidence Distribution ──
    ax5 = fig.add_subplot(gs[1, 1])
    style_ax(ax5, '🔍 Confidence Distribution')

    tb_probs     = probs[labels == 1]
    normal_probs = probs[labels == 0]

    ax5.hist(normal_probs, bins=20, alpha=0.7,
             color=GREEN, label='Normal',
             density=True)
    ax5.hist(tb_probs, bins=20, alpha=0.7,
             color=RED, label='TB',
             density=True)
    ax5.axvline(x=0.5, color=YELLOW,
                linestyle='--', lw=1.5,
                label='Decision boundary')
    ax5.set_xlabel('Predicted TB Probability')
    ax5.set_ylabel('Density')
    ax5.legend(facecolor=CARD, labelcolor=TEXT,
               fontsize=9)
    ax5.grid(True, alpha=0.1, color=BORDER)

    # ── 6. Summary Scorecard ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(CARD)
    ax6.axis('off')
    ax6.set_title('🏆 Model Scorecard',
                  fontsize=12, fontweight='bold',
                  color=ACCENT, pad=12)

    tn, fp, fn, tp = cm.ravel()
    sensitivity    = tp / (tp + fn)
    specificity    = tn / (tn + fp)
    ppv            = tp / (tp + fp)
    npv            = tn / (tn + fn)
    f1             = report_dict['TB']['f1-score']
    accuracy       = (tp + tn) / (tp + tn + fp + fn)

    metrics_data = [
        ('Accuracy',    f'{accuracy*100:.2f}%',  ACCENT),
        ('Sensitivity', f'{sensitivity*100:.2f}%', GREEN),
        ('Specificity', f'{specificity*100:.2f}%', GREEN),
        ('Precision',   f'{ppv*100:.2f}%',        YELLOW),
        ('NPV',         f'{npv*100:.2f}%',        YELLOW),
        ('F1 Score',    f'{f1:.4f}',              ACCENT),
        ('ROC AUC',     f'{roc_auc:.4f}',         ACCENT),
        ('Test Samples',f'{len(labels)}',          TEXT_DIM),
    ]

    for i, (label, value, color) in enumerate(metrics_data):
        y_pos = 0.88 - i * 0.11
        ax6.text(0.05, y_pos, label,
                 transform=ax6.transAxes,
                 color=TEXT_DIM, fontsize=10)
        ax6.text(0.75, y_pos, value,
                 transform=ax6.transAxes,
                 color=color, fontsize=11,
                 fontweight='bold',
                 ha='right')
        line = plt.Line2D(
            [0.02, 0.98],
            [y_pos - 0.03, y_pos - 0.03],
            transform=ax6.transAxes,
            color=BORDER, linewidth=0.5
        )
        ax6.add_line(line)

    # ── Main title ──
    fig.suptitle(
        '🫁 Lung3D AI — Full Model Evaluation Report',
        fontsize=16, fontweight='bold',
        color=TEXT, y=0.98
    )

    # Save
    save_path = Path(save_dir) / 'evaluation_report.png'
    plt.savefig(
        save_path, dpi=120,
        bbox_inches='tight',
        facecolor=BG
    )
    plt.close()
    print(f"\n💾 Evaluation report saved to {save_path}")
    return save_path


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    print("🔍 Running Full Model Evaluation...\n")

    device = get_device()

    # Load data
    print("📦 Loading test data...")
    _, _, test_loader = get_dataloaders(batch_size=16)

    # Load model
    print("🧠 Loading trained model...")
    model      = TBClassifierWithFeatures(num_classes=2)
    checkpoint = torch.load(
        'models/tb_cnn_best.pth',
        map_location=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model      = model.to(device)
    model.eval()
    print("✅ Model loaded!")

    # Get predictions
    print("🔮 Running inference on test set...")
    labels, preds, probs = get_predictions(
        model, test_loader, device
    )

    # Print metrics
    cm, fpr, tpr, roc_auc = print_metrics(
        labels, preds, probs
    )

    # Plot everything
    print("\n📊 Generating evaluation plots...")
    save_path = plot_evaluation(
        labels, preds, probs,
        cm, fpr, tpr, roc_auc
    )

    print(f"\n✅ Evaluation complete!")
    print(f"   Open {save_path} to see the full report")


if __name__ == "__main__":
    main()
    