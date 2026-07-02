"""Inference + Grad-CAM test."""
import sys
from pathlib import Path

# Proje kökünü path'e ekle (src.* importları için)
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
from PIL import Image

from src.defect_classifier.inference import (
    get_inference_engine,
    EQUIPMENT_LIST,
    CLASS_LABELS_TR,
)


def main():
    engine = get_inference_engine()

    print('=' * 70)
    print('INFERENCE + GRAD-CAM TEST')
    print('=' * 70)

    test_dir = Path('data/insplad_fault/organized/test')
    output_dir = Path('outputs/defect_inference_test')
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for equipment in EQUIPMENT_LIST:
        eq_test_dir = test_dir / equipment
        if not eq_test_dir.exists():
            continue

        # Her sınıftan 1 örnek
        for class_dir in sorted(eq_test_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            if not images:
                continue

            test_img_path = sorted(images)[0]

            # Load + predict
            with open(test_img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')

            result = engine.predict_with_gradcam(img, equipment)

            if result:
                # Grad-CAM kaydet
                gradcam_path = output_dir / f'{equipment}_{class_dir.name}_gradcam.jpg'
                if result['gradcam_overlay'] is not None:
                    cv2.imwrite(str(gradcam_path), result['gradcam_overlay'])

                print(f"\n{equipment} / {class_dir.name}:")
                print(f"  Pred: {result['predicted_class']} "
                      f"({result['predicted_class_tr']})")
                print(f"  Confidence: {result['confidence']:.4f}")
                print(f"  Defective: {result['is_defective']}")
                print(f"  Grad-CAM: {gradcam_path}")

                results.append({
                    'equipment': equipment,
                    'true_class': class_dir.name,
                    'predicted': result['predicted_class'],
                    'confidence': result['confidence'],
                    'correct': result['predicted_class'] == class_dir.name,
                })

    # Özet
    print(f"\n{'='*70}")
    print(f"ÖZET: {sum(1 for r in results if r['correct'])}/{len(results)} doğru")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
