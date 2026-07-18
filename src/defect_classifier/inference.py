"""Stage 2 defect classifier inference + Grad-CAM visualization."""
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import cv2

from src.defect_classifier.model import create_defect_classifier


# Equipment → class names mapping (eğitilmiş modellerden)
EQUIPMENT_MODELS_DIR = Path('models/trained/defect_classifiers')

# Post-hoc temperature-scaling calibration (per-equipment T). Optional file —
# absent → T=1.0 everywhere (uncalibrated, original behaviour). See
# src/defect_classifier/calibration.py.
CALIBRATION_PATH = EQUIPMENT_MODELS_DIR / 'calibration.json'

EQUIPMENT_LIST = [
    'glass-insulator',
    'lightning-rod-suspension',
    'polymer-insulator-upper-shackle',
    'vari-grip',
    'yoke-suspension',
]

# YOLO sınıf adı → defect classifier ekipman adı eşleştirme
# (Stage 1 ile Stage 2 arasındaki köprü)
YOLO_TO_DEFECT_EQUIPMENT = {
    'insulator_glass': 'glass-insulator',
    'insulator_polymer': 'polymer-insulator-upper-shackle',
    # Diğer YOLO sınıfları için defect modeli yoksa None
    'insulator_porcelain': None,    # InsPLAD-fault'ta yok
    'insulator_generic': None,
    'tower': None,
    'spacer': None,
    'damper': None,                  # damper-stockbridge/spiral InsPLAD-det'te ama fault'ta farklı
    'tower_id_plate': None,
}

# Türkçe sınıf etiketleri (UI için)
CLASS_LABELS_TR = {
    'good': 'Normal',
    'rust': 'Korozyon (Pas)',
    'missing-cap': 'Kırık/Eksik Kap',
    'bird-nest': 'Kuş Yuvası',
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DefectInferenceEngine:
    """5 ekipman için defect classifier yükleyici + Grad-CAM."""

    def __init__(self, device: str = 'cuda', use_calibration: bool = True):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.models: Dict[str, dict] = {}  # equipment → {model, classes, image_size}

        # Post-hoc calibration: per-equipment temperature. Toggle with the flag so
        # demos can show raw vs calibrated side by side.
        self.use_calibration = use_calibration
        self.temperatures: Dict[str, float] = self._load_calibration()

        # Transform (inference için augmentation YOK)
        self.transform = T.Compose([
            T.Resize((384, 384)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    @staticmethod
    def _load_calibration() -> Dict[str, float]:
        """Load per-equipment temperatures from calibration.json, or {} if absent.

        Backward compatible: a missing/invalid file yields no temperatures, so
        every prediction falls back to T=1.0 (original, uncalibrated behaviour).
        """
        import json
        if not CALIBRATION_PATH.exists():
            return {}
        try:
            data = json.loads(CALIBRATION_PATH.read_text(encoding='utf-8'))
        except Exception:
            return {}
        temps: Dict[str, float] = {}
        for eq, info in data.items():
            if eq.startswith('_'):
                continue  # skip _meta
            try:
                t = float(info['temperature'])
                if t > 0:
                    temps[eq] = t
            except (KeyError, TypeError, ValueError):
                continue
        return temps

    def load_model(self, equipment: str):
        """Belirli bir ekipman modelini yükle (lazy loading)."""
        if equipment in self.models:
            return self.models[equipment]

        ckpt_path = EQUIPMENT_MODELS_DIR / equipment / 'best.pt'
        if not ckpt_path.exists():
            return None

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        model = create_defect_classifier(
            num_classes=ckpt['num_classes'],
            pretrained=False,  # Custom weights yüklüyoruz
        )
        model.load_state_dict(ckpt['model_state'])
        model.to(self.device)
        model.eval()

        self.models[equipment] = {
            'model': model,
            'classes': ckpt['classes'],
            'image_size': ckpt.get('image_size', 384),
        }
        return self.models[equipment]

    def predict(self, image: Image.Image, equipment: str) -> Optional[Dict]:
        """Bir görüntüde defect sınıflandırması yap.

        Returns:
            {
                'equipment': 'glass-insulator',
                'predicted_class': 'rust',
                'predicted_class_tr': 'Korozyon (Pas)',
                'confidence': 0.87,
                'all_probabilities': {'good': 0.13, 'rust': 0.87},
                'is_defective': True,
            }
        """
        model_info = self.load_model(equipment)
        if model_info is None:
            return None

        model = model_info['model']
        classes = model_info['classes']

        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Forward pass -> raw logits (temperature applied post-hoc below).
        with torch.no_grad():
            logits = model(img_tensor)

        # Temperature scaling: calibrated = softmax(logits / T). T>0 is monotonic,
        # so the predicted class is unchanged — only the confidence is corrected.
        temperature = self.temperatures.get(equipment, 1.0) if self.use_calibration else 1.0
        calibrated = bool(self.use_calibration and equipment in self.temperatures)

        raw_probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        cal_probs = F.softmax(logits / temperature, dim=1).squeeze().cpu().numpy()

        # Highest probability class (identical for raw/cal — T is monotonic).
        pred_idx = int(cal_probs.argmax())
        pred_class = classes[pred_idx]

        # Public fields carry the CALIBRATED (honest) confidence; raw kept for
        # transparency/comparison. Existing field names are preserved.
        confidence = float(cal_probs[pred_idx])
        all_probs = {classes[i]: float(cal_probs[i]) for i in range(len(classes))}
        raw_all = {classes[i]: float(raw_probs[i]) for i in range(len(classes))}

        # Defect mı? (good dışındaki her şey defect)
        is_defective = pred_class != 'good'

        return {
            'equipment': equipment,
            'predicted_class': pred_class,
            'predicted_class_tr': CLASS_LABELS_TR.get(pred_class, pred_class),
            'confidence': confidence,
            'all_probabilities': all_probs,
            'is_defective': is_defective,
            # --- calibration transparency (additive; does not break callers) ---
            'calibrated': calibrated,
            'temperature': float(temperature),
            'raw_confidence': float(raw_probs[pred_idx]),
            'raw_probabilities': raw_all,
        }

    def generate_gradcam(
        self,
        image: Image.Image,
        equipment: str,
        target_class_idx: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """Grad-CAM heatmap oluştur.

        Returns:
            (H, W, 3) BGR heatmap overlay görüntüsü
        """
        model_info = self.load_model(equipment)
        if model_info is None:
            return None

        model = model_info['model']
        image_size = model_info['image_size']

        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        img_tensor.requires_grad_()

        # Find target layer (EfficientNetV2-S son conv block)
        target_layer = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module

        if target_layer is None:
            return None

        # Hooks for Grad-CAM
        activations = {}
        gradients = {}

        def forward_hook(module, input, output):
            activations['value'] = output

        def backward_hook(module, grad_input, grad_output):
            gradients['value'] = grad_output[0]

        fh = target_layer.register_forward_hook(forward_hook)
        bh = target_layer.register_full_backward_hook(backward_hook)

        try:
            # Forward + backward
            output = model(img_tensor)

            if target_class_idx is None:
                target_class_idx = int(output.argmax(dim=1).item())

            model.zero_grad()
            class_score = output[0, target_class_idx]
            class_score.backward()

            # Grad-CAM hesapla
            grads = gradients['value'].squeeze().cpu().numpy()       # (C, H, W)
            acts = activations['value'].squeeze().cpu().detach().numpy()  # (C, H, W)

            # Channel-wise weight (global average pool)
            weights = grads.mean(axis=(1, 2))   # (C,)

            # Weighted sum
            cam = np.zeros(acts.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * acts[i]

            # ReLU + normalize
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (image_size, image_size))
            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min())

            # Overlay
            cam_uint8 = np.uint8(255 * cam)
            heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

            # Orijinal görüntü
            img_array = np.array(image.resize((image_size, image_size)))
            if img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Blend (60% heatmap + 40% original)
            overlay = cv2.addWeighted(heatmap, 0.6, img_array, 0.4, 0)

            return overlay

        finally:
            fh.remove()
            bh.remove()

    def predict_with_gradcam(
        self,
        image: Image.Image,
        equipment: str,
    ) -> Optional[Dict]:
        """Predict + Grad-CAM birleşik çıktı.

        Returns:
            predict() çıktısı + 'gradcam_overlay': np.ndarray
        """
        result = self.predict(image, equipment)
        if result is None:
            return None

        model_info = self.load_model(equipment)
        classes = model_info['classes']
        target_idx = classes.index(result['predicted_class'])

        gradcam = self.generate_gradcam(image, equipment, target_class_idx=target_idx)
        result['gradcam_overlay'] = gradcam

        return result


# Singleton pattern - tek instance kullan (RAM tasarrufu)
_inference_engine: Optional[DefectInferenceEngine] = None


def get_inference_engine() -> DefectInferenceEngine:
    """Global inference engine al (lazy initialization)."""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = DefectInferenceEngine()
    return _inference_engine


def predict_defect(image_path: Path, yolo_class: str) -> Optional[Dict]:
    """High-level wrapper: YOLO sınıf adından defect prediction.

    Args:
        image_path: Görüntü dosya yolu (croplanmış ekipman)
        yolo_class: YOLO'nun verdiği sınıf (insulator_glass vb.)

    Returns:
        Predict + Grad-CAM çıktısı veya None
    """
    equipment = YOLO_TO_DEFECT_EQUIPMENT.get(yolo_class)
    if equipment is None:
        return None

    # Unicode-safe loading
    with open(image_path, 'rb') as f:
        image = Image.open(f).convert('RGB')

    engine = get_inference_engine()
    return engine.predict_with_gradcam(image, equipment)
