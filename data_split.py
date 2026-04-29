"""
data_split.py — Create Label Budget Splits for Semi-Supervised Experiments
===========================================================================
Takes a YOLO-format dataset directory and creates 5%, 10%, 20% labeled subsets.
The remaining frames become the "unlabeled" pool.

Usage:
    python data_split.py --data_dir dataset/ --output_dir splits/

Expected dataset/ structure:
    dataset/
        images/
            train/
                frame_000.jpg
                frame_001.jpg
                ...
        labels/
            train/
                frame_000.txt
                frame_001.txt
                ...

If you don't have a YOLO-format dataset yet, this script can also generate one
from your video + model by running inference and saving pseudo-ground-truth:
    python data_split.py --from_video video.mp4 --model best.pt --output_dir splits/
"""

import os
import shutil
import random
import argparse
import cv2
import numpy as np
from pathlib import Path


def extract_frames_and_labels(video_path, model_path, dataset_dir):
    """
    Generate a YOLO-format dataset from video using a trained model.
    The model's predictions on ALL frames serve as ground truth
    (since we have a fully trained model already).
    """
    from ultralytics import YOLO

    dataset_dir = Path(dataset_dir)
    img_dir = dataset_dir / "images" / "train"
    lbl_dir = dataset_dir / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Extracting {total} frames from {video_path}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_name = f"frame_{frame_idx:05d}.png"
        cv2.imwrite(str(img_dir / img_name), frame)

        results = model(frame, verbose=False)
        h, w = frame.shape[:2]

        lbl_name = f"frame_{frame_idx:05d}.txt"
        with open(lbl_dir / lbl_name, "w") as f:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = ((x1 + x2) / 2.0) / w
                cy = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                conf = float(box.conf[0])
                if conf > 0.5:
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{total}")
        frame_idx += 1

    cap.release()
    print(f"  Dataset created: {frame_idx} frames in {dataset_dir}")
    return dataset_dir


def use_existing_dataset(images_dir, labels_dir, dataset_dir):
    """
    Copy an existing YOLO dataset (images/ + labels/) into the expected structure.
    Supports your 500-image labeled dataset with .png images.
    """
    dataset_dir = Path(dataset_dir)
    img_dir = dataset_dir / "images" / "train"
    lbl_dir = dataset_dir / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    n = 0
    for img in sorted(images_dir.glob("*")):
        if img.suffix.lower() in (".png", ".jpg", ".jpeg"):
            shutil.copy2(img, img_dir / img.name)
            lbl_file = labels_dir / f"{img.stem}.txt"
            if lbl_file.exists():
                shutil.copy2(lbl_file, lbl_dir / f"{img.stem}.txt")
            n += 1

    print(f"  Copied {n} images from {images_dir}")
    return dataset_dir


def create_splits(dataset_dir, output_dir, seed=42):
    """
    Create 5%, 10%, 20% labeled splits + corresponding unlabeled pools.
    Also creates a held-out test set (15% of data) for evaluation.
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    img_dir = dataset_dir / "images" / "train"
    lbl_dir = dataset_dir / "labels" / "train"

    # Get all frames that have labels
    all_images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    all_frames = sorted([f.stem for f in all_images
                         if (lbl_dir / f"{f.stem}.txt").exists()])

    print(f"Total annotated frames: {len(all_frames)}")

    random.seed(seed)
    random.shuffle(all_frames)

    # Hold out 15% for testing
    n_test = max(1, int(len(all_frames) * 0.15))
    test_frames = all_frames[:n_test]
    train_pool = all_frames[n_test:]

    print(f"Test set: {n_test} frames")
    print(f"Train pool: {len(train_pool)} frames")

    # Create test set
    _copy_subset(test_frames, img_dir, lbl_dir, output_dir / "test")

    # Create labeled splits
    for pct in [5, 10, 20, 100]:
        n_labeled = max(1, int(len(train_pool) * pct / 100))
        labeled = train_pool[:n_labeled]
        unlabeled = train_pool[n_labeled:]

        split_name = f"labeled_{pct}pct"
        unlabeled_name = f"unlabeled_{pct}pct"

        _copy_subset(labeled, img_dir, lbl_dir, output_dir / split_name)
        _copy_subset(unlabeled, img_dir, lbl_dir, output_dir / unlabeled_name,
                     copy_labels=False)  # unlabeled = images only

        print(f"  {pct}% split: {n_labeled} labeled, {len(unlabeled)} unlabeled")

    # Save frame lists for reproducibility
    _save_list(test_frames, output_dir / "test_frames.txt")
    _save_list(train_pool, output_dir / "train_pool.txt")

    print(f"\nAll splits saved to {output_dir}/")


def _copy_subset(frames, img_dir, lbl_dir, dest_dir, copy_labels=True):
    """Copy image (and optionally label) files for a list of frame stems."""
    dest_img = dest_dir / "images"
    dest_lbl = dest_dir / "labels"
    dest_img.mkdir(parents=True, exist_ok=True)
    if copy_labels:
        dest_lbl.mkdir(parents=True, exist_ok=True)

    for stem in frames:
        # Support both .png and .jpg
        for ext in (".png", ".jpg", ".jpeg"):
            img_file = img_dir / f"{stem}{ext}"
            if img_file.exists():
                shutil.copy2(img_file, dest_img / img_file.name)
                break
        if copy_labels:
            lbl_file = lbl_dir / f"{stem}.txt"
            if lbl_file.exists():
                shutil.copy2(lbl_file, dest_lbl / f"{stem}.txt")


def _save_list(frames, path):
    """Save a list of frame stems to a text file."""
    with open(path, "w") as f:
        for stem in frames:
            f.write(f"{stem}\n")


def create_yaml(output_dir, split_name):
    """Create a YOLO data.yaml for a given split."""
    output_dir = Path(output_dir)
    yaml_path = output_dir / f"{split_name}.yaml"

    train_path = str((output_dir / split_name / "images").resolve())
    test_path = str((output_dir / "test" / "images").resolve())

    with open(yaml_path, "w") as f:
        f.write(f"train: {train_path}\n")
        f.write(f"val: {test_path}\n")
        f.write(f"test: {test_path}\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['bee']\n")

    print(f"  Created {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create label budget splits")
    parser.add_argument("--data_dir", type=str, default="dataset",
                        help="Existing YOLO dataset directory")
    parser.add_argument("--output_dir", type=str, default="splits",
                        help="Output directory for splits")
    parser.add_argument("--from_video", type=str, default=None,
                        help="Generate dataset from video file")
    parser.add_argument("--from_existing", type=str, nargs=2, default=None,
                        metavar=("IMAGES_DIR", "LABELS_DIR"),
                        help="Use existing labeled dataset: --from_existing images/ labels/")
    parser.add_argument("--model", type=str, default="best.pt",
                        help="Model path (used with --from_video)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.from_video:
        extract_frames_and_labels(args.from_video, args.model, args.data_dir)
    elif args.from_existing:
        use_existing_dataset(args.from_existing[0], args.from_existing[1], args.data_dir)

    create_splits(args.data_dir, args.output_dir, args.seed)

    # Create YAML configs for each split
    for pct in [5, 10, 20, 100]:
        create_yaml(args.output_dir, f"labeled_{pct}pct")

    print("\nDone! Next steps:")
    print("  1. Run pseudo_labeling.py for self-training experiments")
    print("  2. Run label_propagation.py for graph-based experiments")
    print("  3. Run evaluate.py to compare all methods")
