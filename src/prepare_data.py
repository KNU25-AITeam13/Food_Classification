"""
AI Hub 한국 음식 이미지 데이터셋 전처리 스크립트

기능:
- 2단계 폴더 구조(대분류/소분류/) → YOLO ImageFolder 형식(train/클래스명/) 평탄화
- GIF 파일 제외 (jpg, jpeg, png만 처리)
- 확장자 소문자 통일
- train/val/test 70:15:15 분할
- 전처리 완료 후 압축 기능
"""

import os
import shutil
import random
import zipfile
from pathlib import Path
from typing import Optional
from tqdm import tqdm


# 초기 테스트용 20개 대중 음식 클래스 (한식)
TEST_CLASSES = [
    "비빔밥",
    "김치찌개",
    "된장찌개",
    "불고기",
    "삼겹살",
    "김밥",
    "라면",
    "짜장면",
    "짬뽕",
    "떡볶이",
    "삼계탕",
    "갈비찜",
    "배추김치",
    "깍두기",
    "잡채",
    "계란말이",
    "파전",
    "물냉면",
    "칼국수",
    "족발",
]

# Food-101 데이터셋 클래스 매핑 (영문 → 한글)
# bibimbap은 한식 비빔밥과 병합됨
FOOD101_CLASS_MAPPING = {
    # 서양 메인 요리
    "pizza": "피자",
    "hamburger": "햄버거",
    "steak": "스테이크",
    "hot_dog": "핫도그",
    "french_fries": "감자튀김",
    # 파스타/면류
    "spaghetti_bolognese": "스파게티",
    "lasagna": "라자냐",
    # 아시안 요리
    "ramen": "라멘",  # 한식 라면과 구분
    "sushi": "초밥",
    "fried_rice": "볶음밥",
    "dumplings": "만두",
    "pad_thai": "팟타이",
    "pho": "쌀국수",
    # 디저트
    "ice_cream": "아이스크림",
    "cheesecake": "치즈케이크",
    "donuts": "도넛",
    "pancakes": "팬케이크",
    "waffles": "와플",
    # 기타
    "caesar_salad": "시저샐러드",
    "tacos": "타코",
    # 병합 클래스 (한식과 동일)
    "bibimbap": "비빔밥",  # 한식 비빔밥에 병합
}

# Food-101 클래스당 최대 이미지 수 (한식 데이터와 균형 맞춤)
FOOD101_MAX_IMAGES_PER_CLASS = 1000

# 지원하는 이미지 확장자 (GIF 제외)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_all_classes(source_dir: Path) -> list[str]:
    """
    원본 데이터셋에서 모든 클래스(소분류) 목록을 추출합니다.
    
    Args:
        source_dir: 원본 데이터셋 경로 (datasets/kfood)
    
    Returns:
        클래스명 리스트
    """
    classes = []
    for category_dir in source_dir.iterdir():
        if category_dir.is_dir() and not category_dir.name.endswith('.zip'):
            for class_dir in category_dir.iterdir():
                if class_dir.is_dir():
                    classes.append(class_dir.name)
    return sorted(classes)


def collect_images(source_dir: Path, class_name: str) -> list[Path]:
    """
    특정 클래스의 모든 유효한 이미지 파일을 수집합니다.
    
    Args:
        source_dir: 원본 데이터셋 경로
        class_name: 클래스명 (소분류)
    
    Returns:
        이미지 파일 경로 리스트
    """
    images = []
    for category_dir in source_dir.iterdir():
        if category_dir.is_dir() and not category_dir.name.endswith('.zip'):
            class_dir = category_dir / class_name
            if class_dir.exists():
                for img_file in class_dir.iterdir():
                    if img_file.suffix.lower() in VALID_EXTENSIONS:
                        images.append(img_file)
    return images


def split_dataset(
    images: list[Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    이미지 리스트를 train/val/test로 분할합니다.
    
    Args:
        images: 이미지 파일 경로 리스트
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율
        seed: 랜덤 시드
    
    Returns:
        (train_images, val_images, test_images) 튜플
    """
    random.seed(seed)
    shuffled = images.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def collect_food101_images(
    source_dir: Path,
    class_name: str,
    max_images: int = FOOD101_MAX_IMAGES_PER_CLASS,
    seed: int = 42
) -> list[Path]:
    """
    Food-101 데이터셋에서 특정 클래스의 이미지를 수집합니다.
    
    Args:
        source_dir: Food-101 데이터셋 경로 (datasets/food_101)
        class_name: 영문 클래스명 (예: pizza, hamburger)
        max_images: 클래스당 최대 이미지 수
        seed: 랜덤 시드 (샘플링용)
    
    Returns:
        이미지 파일 경로 리스트
    """
    images_dir = source_dir / "images" / class_name
    
    if not images_dir.exists():
        print(f"Warning: Food-101 class directory not found: {images_dir}")
        return []
    
    images = []
    for img_file in images_dir.iterdir():
        if img_file.suffix.lower() in VALID_EXTENSIONS:
            images.append(img_file)
    
    # 최대 이미지 수로 샘플링
    if len(images) > max_images:
        random.seed(seed)
        images = random.sample(images, max_images)
    
    return images


def copy_images(
    images: list[Path],
    dest_dir: Path,
    class_name: str,
    desc: str = ""
) -> int:
    """
    이미지 파일들을 대상 디렉토리로 복사합니다.
    
    Args:
        images: 이미지 파일 경로 리스트
        dest_dir: 대상 디렉토리 (예: data/train)
        class_name: 클래스명
        desc: 진행바 설명
    
    Returns:
        복사된 파일 수
    """
    class_dir = dest_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    for i, img_path in enumerate(tqdm(images, desc=desc, leave=False)):
        # 확장자 소문자 통일
        ext = img_path.suffix.lower()
        if ext == ".jpeg":
            ext = ".jpg"
        
        # 새 파일명: 클래스명_번호.확장자
        new_name = f"{class_name}_{i:04d}{ext}"
        dest_path = class_dir / new_name
        
        try:
            shutil.copy2(img_path, dest_path)
            copied += 1
        except Exception as e:
            print(f"Warning: Failed to copy {img_path}: {e}")
    
    return copied


def prepare_dataset(
    source_dir: Path,
    output_dir: Path,
    mode: str = "full",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> dict:
    """
    데이터셋 전처리 메인 함수
    
    Args:
        source_dir: 원본 데이터셋 경로 (datasets/kfood)
        output_dir: 출력 디렉토리 (data/)
        mode: "test" (20개 클래스) 또는 "full" (전체 클래스)
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율
        seed: 랜덤 시드
    
    Returns:
        통계 정보 딕셔너리
    """
    # 출력 디렉토리 초기화
    if output_dir.exists():
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    
    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 클래스 목록 결정
    if mode == "test":
        classes = TEST_CLASSES
        print(f"Test mode: Processing {len(classes)} classes")
    else:
        classes = get_all_classes(source_dir)
        print(f"Full mode: Processing {len(classes)} classes")
    
    stats = {
        "mode": mode,
        "total_classes": len(classes),
        "train_images": 0,
        "val_images": 0,
        "test_images": 0,
        "skipped_classes": [],
    }
    
    for class_name in tqdm(classes, desc="Processing classes"):
        # 이미지 수집
        images = collect_images(source_dir, class_name)
        
        if not images:
            print(f"Warning: No images found for class '{class_name}'")
            stats["skipped_classes"].append(class_name)
            continue
        
        # 데이터 분할
        train_imgs, val_imgs, test_imgs = split_dataset(
            images, train_ratio, val_ratio, seed
        )
        
        # 이미지 복사
        stats["train_images"] += copy_images(
            train_imgs, train_dir, class_name, f"{class_name} train"
        )
        stats["val_images"] += copy_images(
            val_imgs, val_dir, class_name, f"{class_name} val"
        )
        stats["test_images"] += copy_images(
            test_imgs, test_dir, class_name, f"{class_name} test"
        )
    
    # 클래스 목록 저장
    classes_file = output_dir / "classes.txt"
    valid_classes = [c for c in classes if c not in stats["skipped_classes"]]
    with open(classes_file, "w", encoding="utf-8") as f:
        for class_name in valid_classes:
            f.write(f"{class_name}\n")
    
    stats["valid_classes"] = len(valid_classes)
    
    return stats


def prepare_mixed_dataset(
    kfood_source_dir: Path,
    food101_source_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> dict:
    """
    한식(kfood) + Food-101 통합 데이터셋 전처리
    
    Args:
        kfood_source_dir: AI Hub 한식 데이터셋 경로 (datasets/kfood)
        food101_source_dir: Food-101 데이터셋 경로 (datasets/food_101)
        output_dir: 출력 디렉토리 (data/)
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율
        seed: 랜덤 시드
    
    Returns:
        통계 정보 딕셔너리
    """
    # 출력 디렉토리 초기화
    if output_dir.exists():
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    
    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "mode": "mixed",
        "kfood_classes": 0,
        "food101_classes": 0,
        "total_classes": 0,
        "train_images": 0,
        "val_images": 0,
        "test_images": 0,
        "skipped_classes": [],
        "merged_classes": [],  # 병합된 클래스 (bibimbap → 비빔밥)
    }
    
    # 모든 클래스 수집 (한글명 기준)
    all_classes_data = {}  # {한글클래스명: [이미지경로들]}
    
    # 1단계: 한식 데이터 수집
    print("\n[1/2] Processing Korean food (kfood) dataset...")
    for class_name in tqdm(TEST_CLASSES, desc="Korean food"):
        images = collect_images(kfood_source_dir, class_name)
        
        if not images:
            print(f"Warning: No images found for Korean class '{class_name}'")
            stats["skipped_classes"].append(class_name)
            continue
        
        all_classes_data[class_name] = images
        stats["kfood_classes"] += 1
    
    # 2단계: Food-101 데이터 수집
    print("\n[2/2] Processing Food-101 dataset...")
    for eng_class, kor_class in tqdm(FOOD101_CLASS_MAPPING.items(), desc="Food-101"):
        images = collect_food101_images(food101_source_dir, eng_class, seed=seed)
        
        if not images:
            print(f"Warning: No images found for Food-101 class '{eng_class}'")
            stats["skipped_classes"].append(f"food101:{eng_class}")
            continue
        
        # 이미 한글 클래스가 존재하면 병합 (예: bibimbap → 비빔밥)
        if kor_class in all_classes_data:
            existing_count = len(all_classes_data[kor_class])
            all_classes_data[kor_class].extend(images)
            merged_count = len(images)
            stats["merged_classes"].append(
                f"{eng_class} → {kor_class} (+{merged_count} images, total: {existing_count + merged_count})"
            )
            print(f"  Merged: {eng_class} → {kor_class} (+{merged_count} images)")
        else:
            all_classes_data[kor_class] = images
            stats["food101_classes"] += 1
    
    stats["total_classes"] = len(all_classes_data)
    
    # 3단계: 데이터 분할 및 복사
    print(f"\nSplitting and copying {stats['total_classes']} classes...")
    for class_name, images in tqdm(all_classes_data.items(), desc="Copying"):
        # 데이터 분할
        train_imgs, val_imgs, test_imgs = split_dataset(
            images, train_ratio, val_ratio, seed
        )
        
        # 이미지 복사
        stats["train_images"] += copy_images(
            train_imgs, train_dir, class_name, f"{class_name} train"
        )
        stats["val_images"] += copy_images(
            val_imgs, val_dir, class_name, f"{class_name} val"
        )
        stats["test_images"] += copy_images(
            test_imgs, test_dir, class_name, f"{class_name} test"
        )
    
    # 클래스 목록 저장
    classes_file = output_dir / "classes.txt"
    valid_classes = sorted(all_classes_data.keys())
    with open(classes_file, "w", encoding="utf-8") as f:
        for class_name in valid_classes:
            f.write(f"{class_name}\n")
    
    stats["valid_classes"] = len(valid_classes)
    
    return stats


def compress_dataset(data_dir: Path, output_path: Optional[Path] = None) -> Path:
    """
    전처리된 데이터셋을 압축합니다.
    
    Args:
        data_dir: 전처리된 데이터 디렉토리
        output_path: 출력 zip 파일 경로 (기본: data_dir.zip)
    
    Returns:
        생성된 zip 파일 경로
    """
    if output_path is None:
        output_path = data_dir.parent / f"{data_dir.name}.zip"
    
    print(f"Compressing dataset to {output_path}...")
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(data_dir):
            for file in tqdm(files, desc="Compressing", leave=False):
                file_path = Path(root) / file
                arcname = file_path.relative_to(data_dir.parent)
                zipf.write(file_path, arcname)
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Compression complete: {output_path} ({size_mb:.1f} MB)")
    
    return output_path


def main():
    """CLI 엔트리포인트"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AI Hub 한국 음식 데이터셋 전처리"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="datasets/kfood",
        help="원본 데이터셋 경로 (기본: datasets/kfood)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="출력 디렉토리 (기본: data)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "full"],
        default="test",
        help="처리 모드: test (20개 클래스) 또는 full (전체)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="학습 데이터 비율 (기본: 0.7)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="검증 데이터 비율 (기본: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드 (기본: 42)"
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="전처리 후 압축 파일 생성"
    )
    
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return 1
    
    print("=" * 60)
    print("Korean Food Dataset Preparation")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Mode: {args.mode}")
    print(f"Split ratio: {args.train_ratio}/{args.val_ratio}/{1-args.train_ratio-args.val_ratio:.2f}")
    print("=" * 60)
    
    # 데이터셋 전처리
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


if __name__ == "__main__":
    exit(main())
