"""Train all 5 equipment classifiers sequentially."""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Proje kökünü path'e ekle (src.* importları için)
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from src.defect_classifier.train import train_one_equipment


EQUIPMENT_LIST = [
    'glass-insulator',
    'lightning-rod-suspension',
    'polymer-insulator-upper-shackle',
    'vari-grip',
    'yoke-suspension',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--image_size', type=int, default=384)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Device: {device}")
    print(f"🎯 Hedef: Test F1 ≥ 0.90 her ekipman için")

    if device == 'cpu':
        print("❌ HATA: CUDA yok, eğitim çok yavaş olur. Devam edilmiyor.")
        return

    all_results = {}
    total_start = datetime.now()

    for i, equipment in enumerate(EQUIPMENT_LIST, 1):
        print(f"\n{'='*70}")
        print(f"📋 [{i}/{len(EQUIPMENT_LIST)}] {equipment}")
        print(f"{'='*70}")

        try:
            results = train_one_equipment(
                equipment=equipment,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                image_size=args.image_size,
                device=device,
            )
            all_results[equipment] = results
        except Exception as e:
            import traceback
            print(f"❌ FAILED {equipment}:")
            traceback.print_exc()
            all_results[equipment] = {'error': str(e), 'traceback': traceback.format_exc()}

    total_elapsed = (datetime.now() - total_start).total_seconds() / 60

    print(f"\n{'='*70}")
    print(f"📊 TÜM EĞİTİMLER TAMAMLANDI — Toplam: {total_elapsed:.1f} dk")
    print(f"{'='*70}\n")

    print(f"{'Equipment':<40} {'Val F1':<10} {'Test F1':<10} {'Test Acc':<10} {'Hedef':<10}")
    print(f"{'-'*90}")
    for eq, res in all_results.items():
        if 'error' in res:
            print(f"{eq:<40} ❌ {res['error'][:40]}")
        else:
            hedef = '✅' if res['test_f1'] >= 0.90 else '⚠️'
            print(f"{eq:<40} {res['best_val_f1']:<10.4f} {res['test_f1']:<10.4f} {res['test_acc']:<10.4f} {hedef:<10}")

    # Hedef özeti
    successful = sum(1 for r in all_results.values() if 'error' not in r and r.get('test_f1', 0) >= 0.90)
    print(f"\n🎯 Hedef tutturmuş ekipman: {successful}/{len(EQUIPMENT_LIST)}")

    summary_path = Path('outputs/training_logs/defect_classifier/all_results.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"💾 Sonuçlar: {summary_path}")


if __name__ == '__main__':
    main()
