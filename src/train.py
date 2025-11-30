"""
YOLOv11 한국 음식 분류 모델 학습 스크립트

사용법:
    python src/train.py --config config/train_config_test.yaml
    python src/train.py --config config/train_config_full.yaml
"""

import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO


def load_config(config_path: str) -> dict:
    """
    YAML 설정 파일을 로드합니다.
    
    Args:
        config_path: 설정 파일 경로
    
    Returns:
        설정 딕셔너리
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train(config_path: str, resume: bool = False) -> None:
    """
    모델 학습을 실행합니다.
    
    Args:
        config_path: 설정 파일 경로
        resume: 이전 학습 재개 여부
    """
    # 설정 로드
    config = load_config(config_path)
    
    print("=" * 60)
    print("Korean Food Classification Training")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Model: {config.get('model', 'yolo11m-cls.pt')}")
    print(f"Data: {config.get('data', 'data')}")
    print(f"Epochs: {config.get('epochs', 100)}")
    print(f"Batch size: {config.get('batch', 64)}")
    print(f"Image size: {config.get('imgsz', 224)}")
    print(f"Device: {config.get('device', 0)}")
    print("=" * 60)
    
    # 모델 로드
    model_name = config.pop('model', 'yolo11m-cls.pt')
    
    if resume:
        # 마지막 체크포인트에서 재개
        project = config.get('project', 'runs/classify')
        name = config.get('name', 'korean_food')
        last_pt = Path(project) / name / 'weights' / 'last.pt'
        
        if last_pt.exists():
            print(f"Resuming from: {last_pt}")
            model = YOLO(str(last_pt))
        else:
            print(f"No checkpoint found at {last_pt}, starting fresh")
            model = YOLO(model_name)
    else:
        model = YOLO(model_name)
    
    # 데이터 경로 확인
    data_path = Path(config.get('data', 'data'))
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_path}\n"
            "Please run 'python main.py prepare' first."
        )
    
    # train 디렉토리 확인
    train_dir = data_path / 'train'
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Train directory not found: {train_dir}\n"
            "Please run 'python main.py prepare' first."
        )
    
    # 클래스 수 확인
    num_classes = len([d for d in train_dir.iterdir() if d.is_dir()])
    print(f"Number of classes: {num_classes}")
    
    # 학습 실행
    results = model.train(**config)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # 결과 경로 출력
    project = config.get('project', 'runs/classify')
    name = config.get('name', 'korean_food')
    save_dir = Path(project) / name
    
    print(f"Results saved to: {save_dir}")
    print(f"Best model: {save_dir / 'weights' / 'best.pt'}")
    print(f"Last model: {save_dir / 'weights' / 'last.pt'}")
    
    return results


def validate(model_path: str, data_path: str = "data") -> None:
    """
    학습된 모델을 검증합니다.
    
    Args:
        model_path: 모델 가중치 파일 경로
        data_path: 데이터셋 경로
    """
    print("=" * 60)
    print("Model Validation")
    print("=" * 60)
    
    model = YOLO(model_path)
    results = model.val(data=data_path)
    
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    print(f"Top-1 Accuracy: {results.top1:.4f}")
    print(f"Top-5 Accuracy: {results.top5:.4f}")
    
    return results


def main():
    """CLI 엔트리포인트"""
    parser = argparse.ArgumentParser(
        description="YOLOv11 한국 음식 분류 모델 학습"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='명령어')
    
    # train 서브커맨드
    train_parser = subparsers.add_parser('train', help='모델 학습')
    train_parser.add_argument(
        '--config',
        type=str,
        default='config/train_config_test.yaml',
        help='학습 설정 파일 경로'
    )
    train_parser.add_argument(
        '--resume',
        action='store_true',
        help='이전 학습 재개'
    )
    
    # validate 서브커맨드
    val_parser = subparsers.add_parser('validate', help='모델 검증')
    val_parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='모델 가중치 파일 경로'
    )
    val_parser.add_argument(
        '--data',
        type=str,
        default='data',
        help='데이터셋 경로'
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args.config, args.resume)
    elif args.command == 'validate':
        validate(args.model, args.data)
    else:
        # 기본 동작: train
        parser.print_help()


if __name__ == "__main__":
    main()
