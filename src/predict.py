"""
YOLOv11 한국 음식 분류 모델 추론 스크립트

사용법:
    python src/predict.py --model runs/classify/korean_food_test/weights/best.pt --image path/to/image.jpg
    python src/predict.py --model runs/classify/korean_food_full/weights/best.pt --image path/to/food.jpg --top 5
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
from PIL import Image


def predict_single(
    model: YOLO,
    image_path: str,
    top_k: int = 5,
    conf_threshold: float = 0.01
) -> dict:
    """
    단일 이미지에 대해 예측을 수행합니다.
    
    Args:
        model: YOLO 모델
        image_path: 이미지 파일 경로
        top_k: 상위 K개 예측 결과 반환
        conf_threshold: 최소 신뢰도 임계값
    
    Returns:
        예측 결과 딕셔너리
    """
    results = model(image_path, verbose=False)
    result = results[0]
    
    # 확률값 추출
    probs = result.probs
    
    # Top-K 인덱스 및 확률
    top_indices = probs.top5 if top_k >= 5 else probs.top5[:top_k]
    top_confs = probs.top5conf if top_k >= 5 else probs.top5conf[:top_k]
    
    # 클래스명 매핑
    names = result.names
    predictions = []
    
    for idx, conf in zip(top_indices, top_confs):
        idx = int(idx)
        conf = float(conf)
        if conf >= conf_threshold:
            predictions.append({
                "class_id": idx,
                "class_name": names[idx],
                "confidence": conf,
                "confidence_pct": f"{conf * 100:.2f}%"
            })
    
    return {
        "image": str(image_path),
        "top1_class": names[int(probs.top1)],
        "top1_confidence": float(probs.top1conf),
        "predictions": predictions[:top_k]
    }


def predict_batch(
    model: YOLO,
    image_paths: list[str],
    top_k: int = 5
) -> list[dict]:
    """
    여러 이미지에 대해 배치 예측을 수행합니다.
    
    Args:
        model: YOLO 모델
        image_paths: 이미지 파일 경로 리스트
        top_k: 상위 K개 예측 결과 반환
    
    Returns:
        예측 결과 리스트
    """
    results = []
    for img_path in image_paths:
        result = predict_single(model, img_path, top_k)
        results.append(result)
    return results


def print_prediction(pred: dict, verbose: bool = False) -> None:
    """
    예측 결과를 출력합니다.
    
    Args:
        pred: 예측 결과 딕셔너리
        verbose: 상세 출력 여부
    """
    print(f"\n📷 이미지: {pred['image']}")
    print(f"🏆 예측 결과: {pred['top1_class']} ({pred['top1_confidence']*100:.2f}%)")
    
    if verbose and pred['predictions']:
        print("\n📊 Top-K 예측:")
        print("-" * 40)
        for i, p in enumerate(pred['predictions'], 1):
            bar_len = int(p['confidence'] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {i}. {p['class_name']:<15} {bar} {p['confidence_pct']}")


def load_model(model_path: str) -> YOLO:
    """
    모델을 로드합니다.
    
    Args:
        model_path: 모델 가중치 파일 경로
    
    Returns:
        YOLO 모델
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    return model


def find_images(path: str) -> list[str]:
    """
    경로에서 이미지 파일을 찾습니다.
    
    Args:
        path: 파일 또는 디렉토리 경로
    
    Returns:
        이미지 파일 경로 리스트
    """
    path = Path(path)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    if path.is_file():
        return [str(path)]
    elif path.is_dir():
        images = []
        for ext in valid_extensions:
            images.extend(path.glob(f"*{ext}"))
            images.extend(path.glob(f"*{ext.upper()}"))
        return [str(img) for img in sorted(images)]
    else:
        raise FileNotFoundError(f"Path not found: {path}")


def main():
    """CLI 엔트리포인트"""
    parser = argparse.ArgumentParser(
        description="YOLOv11 한국 음식 분류 모델 추론"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="학습된 모델 가중치 파일 경로 (.pt)"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="입력 이미지 파일 또는 디렉토리 경로"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="출력할 상위 예측 수 (기본: 5)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="상세 출력 모드"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="결과 저장 파일 경로 (JSON)"
    )
    
    args = parser.parse_args()
    
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


if __name__ == "__main__":
    exit(main())
