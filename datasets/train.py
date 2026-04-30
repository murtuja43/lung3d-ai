import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from datasets.dataloader import get_dataloaders
from datasets.model import TBClassifier, get_device, print_model_summary

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
CONFIG = {
    'epochs':      20,
    'batch_size':  16,
    'lr':          0.0001,
    'weight_decay': 1e-4,
    'patience':    5,       # Early stopping patience
    'save_dir':    'models',
    'model_name':  'tb_cnn_best.pth',
}


# ─────────────────────────────────────────
# Train one epoch
# ─────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"   Batch {batch_idx+1}/{len(loader)} "
                  f"— Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    accuracy = correct / total * 100
    return avg_loss, accuracy


# ─────────────────────────────────────────
# Validate one epoch
# ─────────────────────────────────────────
def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(device)
            labels  = labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item()
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total * 100
    return avg_loss, accuracy


# ─────────────────────────────────────────
# Plot training curves
# ─────────────────────────────────────────
def plot_training(history, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0a0e1a')

    for ax in [ax1, ax2]:
        ax.set_facecolor('#111827')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e2d45')

    # Loss plot
    ax1.plot(history['train_loss'], color='#00d4ff', label='Train Loss')
    ax1.plot(history['val_loss'],   color='#ff4757', label='Val Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(facecolor='#111827', labelcolor='white')
    ax1.grid(True, alpha=0.2)

    # Accuracy plot
    ax2.plot(history['train_acc'], color='#00ff88', label='Train Acc')
    ax2.plot(history['val_acc'],   color='#ffa502', label='Val Acc')
    ax2.set_title('Training & Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend(facecolor='#111827', labelcolor='white')
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    save_path = Path(save_dir) / 'training_curves.png'
    plt.savefig(save_path, dpi=100,
                bbox_inches='tight',
                facecolor='#0a0e1a')
    plt.close()
    print(f"📊 Training curves saved to {save_path}")


# ─────────────────────────────────────────
# Evaluate on test set
# ─────────────────────────────────────────
def evaluate_test(model, loader, device):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean() * 100

    # Per-class accuracy
    tb_mask     = all_labels == 1
    normal_mask = all_labels == 0

    tb_acc     = (all_preds[tb_mask]     == 1).mean() * 100
    normal_acc = (all_preds[normal_mask] == 0).mean() * 100

    print(f"\n{'='*50}")
    print(f"🧪 TEST SET RESULTS")
    print(f"{'='*50}")
    print(f"   Overall Accuracy : {accuracy:.2f}%")
    print(f"   TB Accuracy      : {tb_acc:.2f}%")
    print(f"   Normal Accuracy  : {normal_acc:.2f}%")
    print(f"{'='*50}")

    return accuracy


# ─────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────
def train():
    print("🚀 Starting TB CNN Training\n")

    # Setup
    device = get_device()
    Path(CONFIG['save_dir']).mkdir(exist_ok=True)

    # Data
    print("\n📦 Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=CONFIG['batch_size']
    )

    # Model
    print("\n🧠 Building model...")
    model = TBClassifier(num_classes=2).to(device)
    print_model_summary(model)

    # Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = CONFIG['lr'],
        weight_decay = CONFIG['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=0.5, patience=3
    )

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  []
    }

    best_val_acc  = 0.0
    patience_count = 0
    save_path     = Path(CONFIG['save_dir']) / CONFIG['model_name']

    print(f"\n🏋️  Training for {CONFIG['epochs']} epochs...\n")

    for epoch in range(1, CONFIG['epochs'] + 1):
        print(f"\n{'─'*50}")
        print(f"Epoch {epoch}/{CONFIG['epochs']}")
        print(f"{'─'*50}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )

        # Scheduler step
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"\n📈 Epoch {epoch} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss  : {val_loss:.4f} | "
              f"Val Acc  : {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch':      epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc':    val_acc,
                'val_loss':   val_loss,
            }, save_path)
            print(f"   💾 New best model saved! Val Acc: {val_acc:.2f}%")
            patience_count = 0
        else:
            patience_count += 1
            print(f"   ⏳ No improvement ({patience_count}/"
                  f"{CONFIG['patience']})")

        # Early stopping
        if patience_count >= CONFIG['patience']:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
            break

    print(f"\n{'='*50}")
    print(f"✅ Training Complete!")
    print(f"   Best Val Accuracy : {best_val_acc:.2f}%")
    print(f"   Model saved to    : {save_path}")
    print(f"{'='*50}")

    # Plot curves
    plot_training(history, CONFIG['save_dir'])

    # Final test evaluation
    print("\n🧪 Loading best model for test evaluation...")
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    evaluate_test(model, test_loader, device)


if __name__ == "__main__":
    train()