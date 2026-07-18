"""Train EfficientNetV2-S defect classifier for one equipment type.

Usage:
    uv run python src/defect_classifier/train.py --equipment glass-insulator
    uv run python src/defect_classifier/train.py --equipment glass-insulator --epochs 50
"""
import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

# Proje kökünü path'e ekle (src.* importları için)
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler  # torch 2.x modern AMP API
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)
import numpy as np
from tqdm import tqdm

from src.defect_classifier.dataset import InsPLADFaultDataset
from src.defect_classifier.model import create_defect_classifier, count_parameters


def train_one_equipment(
    equipment: str,
    epochs: int = 50,
    batch_size: int = 16,        # V2-S 384px, daha küçük batch (RTX 4060 8GB)
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stop_patience: int = 10,
    image_size: int = 384,
    device: str = 'cuda',
    data_root: str = 'data/insplad_fault/organized',   # override for augmented sets
    log_dir: str = 'outputs/training_logs/defect_classifier',
    seed: int = 42,              # determinism (numpy/torch)
) -> dict:
    """Train classifier for one equipment type."""

    # Deterministic seed (data shuffle / init reproducibility)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"\n{'='*70}")
    print(f"🎯 TRAINING: {equipment} (EfficientNetV2-S, {image_size}x{image_size})")
    print(f"{'='*70}\n")

    data_root = Path(data_root)
    output_dir = Path(f'models/trained/defect_classifiers/{equipment}')
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Data root: {data_root}")
    print(f"  Log dir:   {log_dir}")

    # Datasets
    train_ds = InsPLADFaultDataset(data_root, equipment, split='train', image_size=image_size)
    val_ds = InsPLADFaultDataset(data_root, equipment, split='val', image_size=image_size)
    test_ds = InsPLADFaultDataset(data_root, equipment, split='test', image_size=image_size)

    num_classes = len(train_ds.classes)
    print(f"  Classes ({num_classes}): {train_ds.classes}")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    class_weights = train_ds.get_class_weights().to(device)
    print(f"  Class weights: {class_weights.tolist()}")

    # DataLoaders (V2-S 384px için daha düşük num_workers, RAM tasarrufu)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Model
    model = create_defect_classifier(num_classes=num_classes).to(device)
    param_info = count_parameters(model)
    print(f"  Model: EfficientNetV2-S")
    print(f"  Parameters: {param_info['total']:,} ({param_info['total_mb']:.1f} MB)")

    # Optimizer + Scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Mixed precision (V2-S 384px için VRAM tasarrufu)
    scaler = GradScaler('cuda') if device == 'cuda' else None

    best_val_f1 = 0.0
    no_improve_count = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    start_time = time.time()
    epoch = 0

    for epoch in range(1, epochs + 1):
        # ----- Training -----
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [TRAIN]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Mixed precision (FP16) - VRAM tasarrufu + hız
            if scaler is not None:
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_losses.append(loss.item())
            _, preds = outputs.max(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{train_correct/train_total:.3f}")

        train_loss_avg = np.mean(train_losses)
        train_acc = train_correct / train_total

        # ----- Validation -----
        model.eval()
        val_losses = []
        val_preds, val_labels = [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [VAL]", leave=False):
                images, labels = images.to(device), labels.to(device)

                if scaler is not None:
                    with autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_losses.append(loss.item())
                _, preds = outputs.max(1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss_avg = np.mean(val_losses)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss_avg)
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(f"  Epoch {epoch:3d} | LR {current_lr:.6f} | "
              f"Train L {train_loss_avg:.4f} Acc {train_acc:.3f} | "
              f"Val L {val_loss_avg:.4f} Acc {val_acc:.3f} F1 {val_f1:.3f}")

        # Best model save + Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve_count = 0

            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'classes': train_ds.classes,
                'class_to_idx': train_ds.class_to_idx,
                'val_f1': val_f1,
                'val_acc': val_acc,
                'equipment': equipment,
                'architecture': 'tf_efficientnetv2_s.in1k',
                'num_classes': num_classes,
                'image_size': image_size,
            }
            torch.save(checkpoint, output_dir / 'best.pt')
            print(f"    ✅ NEW BEST: F1={val_f1:.4f} saved")
        else:
            no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                print(f"    🛑 Early stopping: no improvement for {early_stop_patience} epochs")
                break

    # ----- Final Test Evaluation -----
    print(f"\n  📊 Loading best model for test evaluation...")
    checkpoint = torch.load(output_dir / 'best.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    test_preds, test_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="TEST"):
            images, labels = images.to(device), labels.to(device)
            if scaler is not None:
                with autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)
            _, preds = outputs.max(1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    cm = confusion_matrix(test_labels, test_preds)
    report = classification_report(
        test_labels, test_preds,
        target_names=train_ds.classes,
        output_dict=True,
        zero_division=0,
    )

    elapsed = time.time() - start_time

    results = {
        'equipment': equipment,
        'classes': train_ds.classes,
        'best_val_f1': best_val_f1,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'training_time_seconds': elapsed,
        'training_time_minutes': elapsed / 60,
        'epochs_trained': epoch,
        'parameters': param_info,
        'image_size': image_size,
        'architecture': 'EfficientNetV2-S',
        'timestamp': datetime.now().isoformat(),
    }

    with open(log_dir / f'{equipment}_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(log_dir / f'{equipment}_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    print(f"\n  ✅ {equipment} COMPLETE")
    print(f"     Best Val F1: {best_val_f1:.4f}")
    print(f"     Test Acc:    {test_acc:.4f}")
    print(f"     Test F1:     {test_f1:.4f}  {'✅ HEDEF AŞILDI (≥0.90)' if test_f1 >= 0.90 else '⚠️  HEDEF ALTINDA'}")
    print(f"     Time:        {elapsed/60:.1f} min")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--equipment', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    train_one_equipment(
        equipment=args.equipment,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        device=args.device,
    )


if __name__ == '__main__':
    main()
