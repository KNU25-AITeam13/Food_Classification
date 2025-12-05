# AGENTS.md

This file provides guidance to AI Agents when working with code in this repository.

## Project Overview

This is a YOLOv11-based food image classification project that supports Korean food (AI Hub dataset) and international food (Food-101 dataset). The project uses Ultralytics YOLOv11m-cls for multi-class food classification.

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv (package manager)
uv sync

# Note: PyTorch installation is platform-specific (see pyproject.toml)
# - Windows: CPU version from PyPI
# - Linux: CUDA 12.8 version from pytorch.org index
```

### Data Preparation
```bash
# Test mode: 20 Korean food classes
uv run python main.py prepare --mode test --compress

# Full mode: 150 Korean food classes
uv run python main.py prepare --mode full --compress

# Mixed mode: 39 classes (20 Korean + 19 Food-101)
uv run python main.py prepare --mode mixed --compress

# Custom split ratios
uv run python main.py prepare --mode test --train-ratio 0.7 --val-ratio 0.15
```

### Training
```bash
# Train with test config (20 classes, 30 epochs)
uv run python main.py train --config config/train_config_test.yaml

# Train with full config (150 classes, 100 epochs)
uv run python main.py train --config config/train_config_full.yaml

# Train with mixed config (39 classes, 100 epochs, optimized)
uv run python main.py train --config config/train_config_mixed.yaml

# Resume training from checkpoint
uv run python main.py train --config config/train_config_test.yaml --resume
```

### Inference
```bash
# Predict single image
uv run python main.py predict --model runs/classify/korean_food_test/weights/best.pt --image food.jpg

# Predict with top-5 results
uv run python main.py predict --model best.pt --image food.jpg -v

# Save results to JSON
uv run python main.py predict --model best.pt --image food.jpg --save results.json

# Predict multiple images from directory
uv run python main.py predict --model best.pt --image /path/to/images/
```

## Architecture

### Project Structure
- **main.py**: CLI entry point with three commands (prepare, train, predict)
- **src/prepare_data.py**: Dataset preprocessing and conversion
- **src/train.py**: Training logic using Ultralytics YOLO
- **src/predict.py**: Inference and prediction utilities
- **config/**: YAML configuration files for different training modes

### Data Pipeline

**Raw Dataset Structure** (AI Hub Korean Food):
```
datasets/kfood/
├── 구이/           # Category level (27 categories)
│   ├── 갈비구이/    # Class level (150 classes total)
│   ├── 불고기/
│   └── ...
├── 국/
│   ├── 미역국/
│   └── ...
└── ...
```

**Processed Structure** (YOLO ImageFolder format):
```
data/
├── train/
│   ├── 비빔밥/      # Class folders
│   ├── 김치찌개/
│   └── ...
├── val/
└── test/
├── classes.txt     # Class name list
```

**Data Split**: 70% train / 15% val / 15% test (customizable)

### Training Modes

1. **Test Mode** (train_config_test.yaml):
   - 20 popular Korean food classes
   - 30 epochs, batch size 64, image size 224
   - Quick pipeline validation
   - Early stopping patience: 10

2. **Full Mode** (train_config_full.yaml):
   - All 150 Korean food classes
   - 100 epochs, batch size 64, image size 224
   - Production training
   - Early stopping patience: 15

3. **Mixed Mode** (train_config_mixed.yaml):
   - 39 classes (20 Korean + 19 Food-101)
   - 100 epochs, batch size 64, image size 320
   - AdamW optimizer, label smoothing 0.1
   - Enhanced data augmentation
   - Early stopping patience: 20
   - Note: `bibimbap` from Food-101 is merged into Korean `비빔밥` class

### Data Preprocessing Details

The `prepare_data.py` module handles:
- **Flattening**: Converts 2-level folder structure (category/class) to flat class folders
- **Image filtering**: Only jpg, jpeg, png (GIF excluded)
- **Extension normalization**: .jpeg → .jpg, all lowercase
- **File renaming**: `클래스명_0000.jpg` format to avoid duplicates
- **Class merging**: Food-101 classes can merge with Korean classes (e.g., bibimbap → 비빔밥)
- **Sampling**: Food-101 images limited to 500 per class for balance (see `FOOD101_MAX_IMAGES_PER_CLASS`)
- **Compression**: Optional zip creation for remote server transfer

### Key Configuration Parameters

Training configs use YAML format with these critical parameters:
- **model**: Pre-trained weight file (yolo11m-cls.pt)
- **data**: Path to processed dataset directory
- **epochs**: Training epochs
- **batch**: Batch size (64 recommended for 16GB VRAM)
- **imgsz**: Input image size (224 for test/full, 320 for mixed)
- **device**: GPU device ID (0 for first GPU)
- **project/name**: Output directory structure (runs/classify/{name})
- **patience**: Early stopping patience

### Food-101 Integration

The project supports integrating Food-101 dataset with Korean food in mixed mode:
- Mapping defined in `FOOD101_CLASS_MAPPING` in prepare_data.py
- English class names → Korean names (e.g., "pizza" → "피자")
- Overlapping classes are merged (e.g., bibimbap merges into 비빔밥)
- Class-wise sampling to balance dataset sizes
- Food-101 source: `datasets/food_101/images/{class_name}/`

### Model Output

After training, results are saved to:
```
runs/classify/{experiment_name}/
├── weights/
│   ├── best.pt      # Best model by validation accuracy
│   └── last.pt      # Latest checkpoint (for resume)
├── results.csv      # Training metrics
└── ...
```

## Important Notes

### Remote GPU Training Workflow
When training on cloud GPU servers with Korean characters:
1. Run `uv run python main.py prepare --mode {mode} --compress` locally
2. Upload `data.zip` to server
3. Extract with `unzip -O cp949 data.zip` or `unar data.zip` (handles Korean filenames correctly)
4. Run `uv sync` on server to install dependencies
5. Start training with appropriate config

### Character Encoding
- All Korean class names use UTF-8 encoding
- `classes.txt` is saved with UTF-8 encoding
- Zip extraction requires cp949 codec on some systems (`unzip -O cp949`)

### Resume Training
- Use `--resume` flag with the same config
- Script automatically finds `{project}/{name}/weights/last.pt`
- If checkpoint doesn't exist, starts fresh training

### Inference Output Format
Prediction results contain:
- **top1_class**: Most confident prediction
- **top1_confidence**: Confidence score (0-1)
- **predictions**: List of top-K predictions with class_id, class_name, confidence

### Test Classes (20)
비빔밥, 김치찌개, 된장찌개, 불고기, 삼겹살, 김밥, 라면, 짜장면, 짬뽕, 떡볶이, 삼계탕, 갈비찜, 배추김치, 깍두기, 잡채, 계란말이, 파전, 물냉면, 칼국수, 족발

### Food-101 Classes Added in Mixed Mode (19)
피자, 햄버거, 스테이크, 핫도그, 감자튀김, 스파게티, 라자냐, 라멘, 초밥, 볶음밥, 만두, 팟타이, 쌀국수, 아이스크림, 치즈케이크, 도넛, 팬케이크, 와플, 시저샐러드, 타코

## Hardware Requirements

- **GPU**: RTX 4080 (16GB VRAM) or equivalent recommended
- **RAM**: 32GB+ for full dataset processing
- **Storage**: 50GB+ (raw datasets + processed data + model outputs)

## Dependency Management

This project uses `uv` instead of pip for faster dependency resolution. The `pyproject.toml` includes platform-specific PyTorch installation:
- Linux systems automatically get CUDA 12.8 builds from pytorch.org
- Windows systems get CPU builds from PyPI by default

Main dependencies:
- ultralytics (YOLO framework)
- torch, torchvision (deep learning)
- pillow (image processing)
- tqdm (progress bars)
- pyyaml (config parsing)
