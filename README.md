# 🍱 Food Classification

YOLOv11을 활용한 다국적 음식 이미지 분류 프로젝트

## 📋 프로젝트 개요

AI Hub 한국 음식 이미지와 Food-101 데이터셋을 활용하여 다양한 음식을 분류하는 딥러닝 모델입니다.

- **모델**: YOLOv11m-cls (Classification)
- **데이터셋**: 
  - AI Hub 한국 음식 이미지 (150개 클래스, 약 15만 장)
  - Food-101 (101개 클래스, 약 10만 장) - 선택적 통합
- **프레임워크**: Ultralytics, PyTorch

## 🏗️ 프로젝트 구조

```
Food_Classification/
├── main.py                          # 메인 CLI (prepare, train, predict)
├── pyproject.toml                   # 의존성 설정 (uv)
├── config/
│   ├── train_config_test.yaml       # 테스트 학습 설정 (한식 20개 클래스)
│   ├── train_config_full.yaml       # 전체 학습 설정 (한식 150개 클래스)
│   └── train_config_mixed.yaml      # 통합 학습 설정 (한식+Food-101 39개 클래스)
├── src/
│   ├── train.py                     # 학습 스크립트
│   ├── predict.py                   # 추론 스크립트
│   └── prepare_data.py              # 데이터 전처리
├── data/                            # 전처리된 데이터 (gitignore)
├── datasets/                        # AI Hub 원본 데이터 (gitignore)
└── runs/                            # 학습 결과 (gitignore)
```

## 🚀 설치 및 환경 설정

### 1. 저장소 클론
```bash
git clone https://github.com/KNU25-AITeam13/Food_Classification.git
cd Food_Classification
```

### 2. 의존성 설치 (uv 사용)
```bash
# uv 설치 (없는 경우)
pip install uv

# 의존성 설치
uv sync
```

> **Note**: Windows에서는 CPU 버전, Linux(클라우드 GPU 서버)에서는 CUDA 12.8 버전 PyTorch가 자동 설치됩니다.

### 3. 데이터셋 준비

#### 한식 데이터셋 (필수)
[AI Hub](https://aihub.or.kr/)에서 "한국 음식 이미지" 데이터셋을 다운로드하여 `datasets/kfood/` 폴더에 압축 해제합니다.

#### Food-101 데이터셋 (선택 - mixed 모드용)
[Kaggle Food-101](https://www.kaggle.com/datasets/dansbecker/food-101) 또는 [공식 사이트](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)에서 다운로드하여 `datasets/food_101/` 폴더에 압축 해제합니다.

## 📖 사용법

### 1️⃣ 데이터 전처리

원본 데이터를 YOLO 학습 형식으로 변환합니다.

```bash
# 한식 테스트용 (20개 클래스) + 압축
uv run python main.py prepare --mode test --compress

# 한식 전체 (150개 클래스) + 압축
uv run python main.py prepare --mode full --compress

# 🆕 한식 + Food-101 통합 (39개 클래스) + 압축
uv run python main.py prepare --mode mixed --compress
```

**한식 20개 클래스:**
비빔밥, 김치찌개, 된장찌개, 불고기, 삼겹살, 김밥, 라면, 짜장면, 짬뽕, 떡볶이, 삼계탕, 갈비찜, 배추김치, 깍두기, 잡채, 계란말이, 파전, 물냉면, 칼국수, 족발

**Food-101 추가 19개 클래스 (mixed 모드):**
피자, 햄버거, 스테이크, 핫도그, 감자튀김, 스파게티, 라자냐, 라멘, 초밥, 볶음밥, 만두, 팟타이, 쌀국수, 아이스크림, 치즈케이크, 도넛, 팬케이크, 와플, 시저샐러드, 타코

> **Note**: `bibimbap`(Food-101)은 한식 `비빔밥`에 병합되어 총 39개 클래스가 됩니다.

### 2️⃣ 모델 학습

#### 원격 GPU 서버에서 학습 시
```bash
# 1. data.zip 업로드 후 압축 해제 (한글 인코딩 문제 해결)
unzip -O cp949 data.zip
# 또는 unar 사용 (더 안정적)
sudo apt install unar && unar data.zip

# 2. 의존성 설치
uv sync

# 3. 학습 시작
uv run python main.py train --config config/train_config_test.yaml
```

#### 학습 명령어
```bash
# 테스트 학습 (한식 20개 클래스, epochs=30)
uv run python main.py train --config config/train_config_test.yaml

# 전체 학습 (한식 150개 클래스, epochs=100)
uv run python main.py train --config config/train_config_full.yaml

# 🆕 통합 학습 (한식+Food-101 39개 클래스, epochs=50)
uv run python main.py train --config config/train_config_mixed.yaml

# 이전 학습 재개
uv run python main.py train --config config/train_config_full.yaml --resume
```

### 3️⃣ 추론

```bash
# 단일 이미지 분류
uv run python main.py predict --model runs/classify/korean_food_test/weights/best.pt --image path/to/food.jpg

# 상세 출력 (Top-5)
uv run python main.py predict --model runs/classify/korean_food_test/weights/best.pt --image path/to/food.jpg -v

# 결과 JSON 저장
uv run python main.py predict --model runs/classify/korean_food_test/weights/best.pt --image path/to/food.jpg --save results.json
```

## ⚙️ 학습 설정

### 테스트 학습 (`train_config_test.yaml`)
| 항목 | 값 |
|------|-----|
| 모델 | yolo11m-cls.pt |
| 클래스 수 | 20 |
| Epochs | 30 |
| Batch Size | 64 |
| Image Size | 224 |
| Early Stopping | patience=10 |

### 전체 학습 (`train_config_full.yaml`)
| 항목 | 값 |
|------|-----|
| 모델 | yolo11m-cls.pt |
| 클래스 수 | 150 |
| Epochs | 100 |
| Batch Size | 64 |
| Image Size | 224 |
| Early Stopping | patience=15 |

### 통합 학습 (`train_config_mixed.yaml`) 🆕
| 항목 | 값 |
|------|-----|
| 모델 | yolo11m-cls.pt |
| 클래스 수 | 39 (한식 20 + Food-101 19) |
| Epochs | 100 |
| Batch Size | 64 |
| Image Size | 320 |
| Early Stopping | patience=20 |
| Label Smoothing | 0.1 |
| 데이터 증강 | 강화 (회전, 색상변환 등) |

## 🖥️ 권장 하드웨어

- **GPU**: RTX 4080 (16GB VRAM) 이상
- **RAM**: 32GB 이상
- **Storage**: 50GB 이상 (데이터셋 + 모델)

## 📊 데이터셋 구조

AI Hub 원본 데이터는 2단계 폴더 구조(대분류/소분류)로 되어 있으며, 전처리 후 YOLO ImageFolder 형식으로 변환됩니다.

```
# 원본 (datasets/kfood/)
대분류(27개)/
├── 구이/ → 갈비구이, 불고기, 삼겹살, ...
├── 국/ → 미역국, 육개장, ...
├── 밥/ → 비빔밥, 김밥, ...
└── ...

# 전처리 후 (data/)
├── train/
│   ├── 비빔밥/
│   ├── 김치찌개/
│   └── ...
├── val/
└── test/
```

**데이터 분할 비율**: Train 70% / Val 15% / Test 15%

## 🔗 참고 자료

- [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/)
- [AI Hub 한국 음식 이미지](https://aihub.or.kr/)
- [Food-101 Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- [PyTorch](https://pytorch.org/)

## 👥 팀

**KNU AI Team 13** - 경북대학교 인공지능 팀 프로젝트

## 📄 라이선스

This project is for educational purposes.