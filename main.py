"""
Korean Food Classification using YOLOv11

한국 음식 이미지 분류 프로젝트 메인 실행 파일
AI Hub 한국 음식 이미지 데이터셋을 활용하여 150개 클래스 분류

사용법:
    # 데이터 전처리 (테스트: 20개 클래스)
    python main.py prepare --mode test --compress
    
    # 데이터 전처리 (전체: 150개 클래스)
    python main.py prepare --mode full --compress
    
    # 모델 학습 (테스트)
    python main.py train --config config/train_config_test.yaml
    
    # 모델 학습 (전체)
    python main.py train --config config/train_config_full.yaml
    
    # 추론
    python main.py predict --model runs/classify/korean_food_test/weights/best.pt --image path/to/image.jpg
"""

import argparse
import sys
from pathlib import Path


def cmd_prepare(args):
    """데이터 전처리 명령어"""
    from src.prepare_data import prepare_dataset, prepare_mixed_dataset, compress_dataset
    
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return 1
    
    print("=" * 60)
    print("Food Dataset Preparation")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Mode: {args.mode}")
    print(f"Split ratio: {args.train_ratio}/{args.val_ratio}/{1-args.train_ratio-args.val_ratio:.2f}")
    print("=" * 60)
    
    # 데이터셋 전처리
    if args.mode == "mixed":
        # 혼합 모드: 한식 + Food-101
        food101_dir = Path("datasets/food_101")
        if not food101_dir.exists():
            print(f"Error: Food-101 directory not found: {food101_dir}")
            return 1
        
        print(f"Food-101 Source: {food101_dir}")
        
        stats = prepare_mixed_dataset(
            kfood_source_dir=source_dir,
            food101_source_dir=food101_dir,
            output_dir=output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed
        )
        
        # 혼합 모드 결과 출력
        print("\n" + "=" * 60)
        print("Preparation Complete!")
        print("=" * 60)
        print(f"Mode: {stats['mode']}")
        print(f"Korean food classes: {stats['kfood_classes']}")
        print(f"Food-101 classes: {stats['food101_classes']}")
        print(f"Total classes: {stats['total_classes']}")
        print(f"Valid classes: {stats['valid_classes']}")
        print(f"Train images: {stats['train_images']:,}")
        print(f"Val images: {stats['val_images']:,}")
        print(f"Test images: {stats['test_images']:,}")
        print(f"Total images: {stats['train_images'] + stats['val_images'] + stats['test_images']:,}")
        
        if stats["merged_classes"]:
            print(f"\nMerged classes ({len(stats['merged_classes'])}):")
            for merged in stats["merged_classes"]:
                print(f"  - {merged}")
        
        if stats["skipped_classes"]:
            print(f"\nSkipped classes ({len(stats['skipped_classes'])}): {stats['skipped_classes']}")
    else:
        # 기존 모드: test 또는 full
        stats = prepare_dataset(
            source_dir=source_dir,
            output_dir=output_dir,
            mode=args.mode,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed
        )
        
        # 결과 출력
        print("\n" + "=" * 60)
        print("Preparation Complete!")
        print("=" * 60)
        print(f"Mode: {stats['mode']}")
        print(f"Total classes: {stats['total_classes']}")
        print(f"Valid classes: {stats['valid_classes']}")
        print(f"Train images: {stats['train_images']:,}")
        print(f"Val images: {stats['val_images']:,}")
        print(f"Test images: {stats['test_images']:,}")
        print(f"Total images: {stats['train_images'] + stats['val_images'] + stats['test_images']:,}")
        
        if stats["skipped_classes"]:
            print(f"\nSkipped classes ({len(stats['skipped_classes'])}): {stats['skipped_classes']}")
    
    # 압축
    if args.compress:
        print("\n" + "-" * 60)
        compress_dataset(output_dir)
    
    print("\nDone!")
    return 0


def cmd_train(args):
    """모델 학습 명령어"""
    from src.train import train
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1
    
    train(str(config_path), args.resume)
    return 0


def cmd_predict(args):
    """추론 명령어"""
    from src.predict import load_model, find_images, predict_single, print_prediction
    
    # 모델 로드
    model = load_model(args.model)
    
    # 이미지 찾기
    image_paths = find_images(args.image)
    print(f"Found {len(image_paths)} image(s)")
    
    if not image_paths:
        print("No images found!")
        return 1
    
    print("=" * 60)
    print("Korean Food Classification")
    print("=" * 60)
    
    # 예측 수행
    results = []
    for img_path in image_paths:
        pred = predict_single(model, img_path, args.top)
        results.append(pred)
        print_prediction(pred, verbose=args.verbose or args.top > 1)
    
    # 결과 저장
    if args.save:
        import json
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Results saved to: {save_path}")
    
    print("\n" + "=" * 60)
    print("Done!")
    return 0


def main():
    """메인 CLI 엔트리포인트"""
    parser = argparse.ArgumentParser(
        description="Food Classification using YOLOv11",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  %(prog)s prepare --mode test --compress    # 한식 20개 클래스 전처리 + 압축
  %(prog)s prepare --mode full --compress    # 한식 150개 클래스 전처리 + 압축
  %(prog)s prepare --mode mixed --compress   # 한식 + Food-101 통합 (39개 클래스) + 압축
  %(prog)s train --config config/train_config_test.yaml    # 테스트 학습
  %(prog)s train --config config/train_config_full.yaml    # 전체 학습
  %(prog)s train --config config/train_config_mixed.yaml   # 통합 학습
  %(prog)s predict --model best.pt --image food.jpg        # 추론
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='명령어')
    
    # ========== prepare 서브커맨드 ==========
    prepare_parser = subparsers.add_parser(
        'prepare',
        help='데이터셋 전처리',
        description='AI Hub 한식 및 Food-101 데이터셋을 YOLO 학습 형식으로 변환합니다.'
    )
    prepare_parser.add_argument(
        '--source',
        type=str,
        default='datasets/kfood',
        help='한식 데이터셋 경로 (기본: datasets/kfood)'
    )
    prepare_parser.add_argument(
        '--output',
        type=str,
        default='data',
        help='출력 디렉토리 (기본: data)'
    )
    prepare_parser.add_argument(
        '--mode',
        type=str,
        choices=['test', 'full', 'mixed'],
        default='test',
        help='처리 모드: test (한식 20개), full (한식 150개), mixed (한식+Food-101 39개)'
    )
    prepare_parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='학습 데이터 비율 (기본: 0.7)'
    )
    prepare_parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='검증 데이터 비율 (기본: 0.15)'
    )
    prepare_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='랜덤 시드 (기본: 42)'
    )
    prepare_parser.add_argument(
        '--compress',
        action='store_true',
        help='전처리 후 압축 파일 생성'
    )
    
    # ========== train 서브커맨드 ==========
    train_parser = subparsers.add_parser(
        'train',
        help='모델 학습',
        description='YOLOv11 분류 모델을 학습합니다.'
    )
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
    
    # ========== predict 서브커맨드 ==========
    predict_parser = subparsers.add_parser(
        'predict',
        help='이미지 분류 추론',
        description='학습된 모델로 음식 이미지를 분류합니다.'
    )
    predict_parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='학습된 모델 가중치 파일 경로 (.pt)'
    )
    predict_parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='입력 이미지 파일 또는 디렉토리 경로'
    )
    predict_parser.add_argument(
        '--top',
        type=int,
        default=5,
        help='출력할 상위 예측 수 (기본: 5)'
    )
    predict_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='상세 출력 모드'
    )
    predict_parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='결과 저장 파일 경로 (JSON)'
    )
    
    # 인자 파싱
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # 명령어 실행
    if args.command == 'prepare':
        return cmd_prepare(args)
    elif args.command == 'train':
        return cmd_train(args)
    elif args.command == 'predict':
        return cmd_predict(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

