# Trained Models

This directory contains the trained model weights for food classification.

## Available Models

### best_mixed_food_v1.pt

**Model Architecture**: YOLOv11l-cls (YOLOv11 Large Classification)

**Dataset**: Mixed Korean + International Food
- 20 Korean food classes (AI Hub)
- 19 Food-101 classes
- Total: 39 classes

**Training Configuration**:
- Epochs: 100
- Batch size: 96
- Image size: 320x320
- Optimizer: AdamW
- Learning rate: 0.001 → 0.01 (cosine decay)
- Label smoothing: 0.1
- Dropout: 0.2
- Early stopping patience: 20

**Performance Metrics**:
- Top-1 Accuracy: **93.65%**
- Top-5 Accuracy: **98.86%**
- Validation Loss: 0.372
- Training Loss: 0.012

**Training Date**: 2025-12-01

**Data Augmentation**:
- HSV color jittering
- Random rotation (±15°)
- Translation (15%)
- Scale (60%)
- Horizontal flip (50%)
- Random erasing (10%)

**Hardware Used**:
- GPU: NVIDIA GPU with 16GB+ VRAM
- Training time: ~7845 seconds (~2.2 hours)

## Usage

### Prediction

```bash
# Single image prediction
uv run python main.py predict --model models/best_mixed_food_v1.pt --image food.jpg

# Verbose output with top-5 results
uv run python main.py predict --model models/best_mixed_food_v1.pt --image food.jpg -v

# Save results to JSON
uv run python main.py predict --model models/best_mixed_food_v1.pt --image food.jpg --save results.json
```

### Python API

```python
from ultralytics import YOLO

# Load model
model = YOLO('models/best_mixed_food_v1.pt')

# Run prediction
results = model.predict('food.jpg', verbose=False)

# Get predictions
probs = results[0].probs
top1_idx = probs.top1
confidence = probs.top1conf.item()
class_name = model.names[top1_idx]

print(f"Prediction: {class_name} ({confidence:.2%})")
```

## Supported Classes

### Korean Food (20)
비빔밥, 김치찌개, 된장찌개, 불고기, 삼겹살, 김밥, 라면, 짜장면, 짬뽕, 떡볶이, 삼계탕, 갈비찜, 배추김치, 깍두기, 잡채, 계란말이, 파전, 물냉면, 칼국수, 족발

### International Food (19)
피자, 햄버거, 스테이크, 핫도그, 감자튀김, 스파게티, 라자냐, 라멘, 초밥, 볶음밥, 만두, 팟타이, 쌀국수, 아이스크림, 치즈케이크, 도넛, 팬케이크, 와플, 시저샐러드

## Model Size

- File size: 25 MB
- Parameters: ~25M (YOLOv11l backbone)

## Notes

- This model is trained on mixed Korean and international food dataset
- Best validation accuracy achieved at epoch ~76
- Model uses label smoothing to improve generalization
- Supports batch inference for faster processing of multiple images
