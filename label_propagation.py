"""
label_propagation.py — Graph-Based Semi-Supervised Frame Classification
=========================================================================
Implements label propagation for bee detection using:
  - Gaussian Harmonic Functions (GHF) via sklearn LabelPropagation
  - Local & Global Consistency (LGC) via sklearn LabelSpreading

The approach:
  1. Extract feature embeddings from each frame using YOLO backbone
  2. Build a k-NN similarity graph over labeled + unlabeled frames
  3. Propagate frame-level labels (bee count bins) from labeled to unlabeled
  4. Use propagated labels to select informative unlabeled frames for training

Usage:
    python label_propagation.py --splits_dir splits/ --label_pct 10 --k 10

Outputs:
    results/label_propagation/
        embeddings.npy          — frame embeddings
        ghf_predictions.csv     — GHF propagated labels
        lgc_predictions.csv     — LGC propagated labels
        selected_frames.txt     — frames selected for training augmentation
        metrics.csv             — evaluation metrics
"""

import os
import shutil
import csv
import argparse
import numpy as np
from pathlib import Path
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import accuracy_score, classification_report
import cv2


def extract_embeddings(image_dir, model_path):
    """
    Extract feature embeddings from frames using YOLO detection statistics.
    Returns (embeddings, frame_stems) where embeddings is (N, D) array.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    image_dir = Path(image_dir)
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    img_files = sorted([f for f in image_dir.glob("*") if f.suffix.lower() in IMG_EXTS])

    print(f"Extracting embeddings from {len(img_files)} frames...")

    embeddings = []
    stems = []

    for i, img_path in enumerate(img_files):
        # Use detection-based features as embedding
        # This captures: number of bees, confidence, spatial distribution, sizes
        emb = _detection_features(model, str(img_path))
        embeddings.append(emb)
        stems.append(img_path.stem)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(img_files)} frames processed")

    embeddings = np.array(embeddings)
    print(f"  Embeddings shape: {embeddings.shape}")
    return embeddings, stems


def _detection_features(model, img_path):
    """
    Fallback embedding: run detection and create a feature vector from
    detection statistics (count, avg confidence, spatial distribution).
    """
    results = model(img_path, verbose=False)
    boxes = results[0].boxes
    img = results[0].orig_img
    h, w = img.shape[:2]

    n_detections = len(boxes)

    if n_detections == 0:
        return np.zeros(12)

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    # Normalized centers
    centers_x = ((xyxy[:, 0] + xyxy[:, 2]) / 2.0) / w
    centers_y = ((xyxy[:, 1] + xyxy[:, 3]) / 2.0) / h

    # Box sizes
    widths = (xyxy[:, 2] - xyxy[:, 0]) / w
    heights = (xyxy[:, 3] - xyxy[:, 1]) / h

    features = [
        n_detections,
        confs.mean(), confs.std() if len(confs) > 1 else 0,
        centers_x.mean(), centers_x.std() if len(centers_x) > 1 else 0,
        centers_y.mean(), centers_y.std() if len(centers_y) > 1 else 0,
        widths.mean(), widths.std() if len(widths) > 1 else 0,
        heights.mean(), heights.std() if len(heights) > 1 else 0,
        # Pairwise distance (dispersion) if multiple bees
        _mean_pairwise_dist(centers_x, centers_y)
    ]

    return np.array(features)


def _mean_pairwise_dist(cx, cy):
    """Mean pairwise Euclidean distance between detection centers."""
    if len(cx) < 2:
        return 0.0
    dists = []
    for i in range(len(cx)):
        for j in range(i + 1, len(cx)):
            d = np.sqrt((cx[i] - cx[j])**2 + (cy[i] - cy[j])**2)
            dists.append(d)
    return np.mean(dists)


def create_frame_labels(image_dir, label_dir):
    """
    Create frame-level labels based on bee count in each frame.
    Labels: 0 = no bees, 1 = 1-2 bees, 2 = 3+ bees
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    labels = {}
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    for img_path in sorted([f for f in image_dir.glob("*") if f.suffix.lower() in IMG_EXTS]):
        lbl_path = label_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            with open(lbl_path) as f:
                n_bees = len([l for l in f.readlines() if l.strip()])
        else:
            n_bees = 0

        # Bin into classes
        if n_bees == 0:
            labels[img_path.stem] = 0
        elif n_bees <= 2:
            labels[img_path.stem] = 1
        else:
            labels[img_path.stem] = 2

    return labels


def run_label_propagation(splits_dir, label_pct, k, alpha, model_path):
    """
    Run GHF and LGC label propagation and evaluate.
    """
    splits_dir = Path(splits_dir)
    results_dir = Path(f"results/label_propagation_{label_pct}pct")
    results_dir.mkdir(parents=True, exist_ok=True)

    labeled_dir = splits_dir / f"labeled_{label_pct}pct"
    unlabeled_dir = splits_dir / f"unlabeled_{label_pct}pct"
    full_labeled_dir = splits_dir / "labeled_100pct"  # for ground truth

    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    # ── Collect all frames ──
    labeled_imgs = sorted([f for f in (labeled_dir / "images").iterdir() if f.suffix.lower() in IMG_EXTS])
    unlabeled_imgs = sorted([f for f in (unlabeled_dir / "images").iterdir() if f.suffix.lower() in IMG_EXTS])
    all_stems = [f.stem for f in labeled_imgs] + [f.stem for f in unlabeled_imgs]

    print(f"Labeled frames: {len(labeled_imgs)}")
    print(f"Unlabeled frames: {len(unlabeled_imgs)}")

    # ── Extract embeddings for all frames ──
    # Combine all images into one directory for batch extraction
    all_img_dir = results_dir / "all_images"
    all_img_dir.mkdir(exist_ok=True)

    for img in labeled_imgs:
        shutil.copy2(img, all_img_dir / img.name)
    for img in unlabeled_imgs:
        if not (all_img_dir / img.name).exists():
            shutil.copy2(img, all_img_dir / img.name)

    embeddings, stems = extract_embeddings(all_img_dir, model_path)
    np.save(results_dir / "embeddings.npy", embeddings)

    # ── Create label arrays ──
    # Get ground truth labels for labeled frames
    labeled_labels = create_frame_labels(
        labeled_dir / "images", labeled_dir / "labels")

    # Get ground truth for ALL frames (for evaluation)
    gt_labels = create_frame_labels(
        full_labeled_dir / "images", full_labeled_dir / "labels")

    # Build label array: known for labeled, -1 for unlabeled
    labeled_stems = set(f.stem for f in labeled_imgs)
    y = np.array([
        labeled_labels.get(s, -1) if s in labeled_stems else -1
        for s in stems
    ])

    y_true = np.array([gt_labels.get(s, 0) for s in stems])

    n_labeled = (y != -1).sum()
    n_unlabeled = (y == -1).sum()
    print(f"\nLabel array: {n_labeled} labeled, {n_unlabeled} unlabeled")

    # ── Run GHF (LabelPropagation) ──
    print(f"\nRunning GHF (LabelPropagation) with k={k}...")
    ghf = LabelPropagation(kernel='knn', n_neighbors=k, max_iter=1000)
    ghf.fit(embeddings, y)
    y_ghf = ghf.predict(embeddings)

    # Evaluate on unlabeled frames only (where we propagated)
    unlabeled_mask = y == -1
    ghf_acc = accuracy_score(y_true[unlabeled_mask], y_ghf[unlabeled_mask])
    print(f"  GHF accuracy on unlabeled frames: {ghf_acc:.4f}")

    # ── Run LGC (LabelSpreading) ──
    print(f"\nRunning LGC (LabelSpreading) with k={k}, alpha={alpha}...")
    lgc = LabelSpreading(kernel='knn', n_neighbors=k, alpha=alpha, max_iter=1000)
    lgc.fit(embeddings, y)
    y_lgc = lgc.predict(embeddings)

    lgc_acc = accuracy_score(y_true[unlabeled_mask], y_lgc[unlabeled_mask])
    print(f"  LGC accuracy on unlabeled frames: {lgc_acc:.4f}")

    # ── Save predictions ──
    _save_predictions(stems, y_ghf, results_dir / "ghf_predictions.csv")
    _save_predictions(stems, y_lgc, results_dir / "lgc_predictions.csv")

    # ── Select high-confidence frames for training augmentation ──
    # Use LGC probabilities to select frames where propagation is confident
    lgc_probs = lgc.label_distributions_
    lgc_confidence = lgc_probs.max(axis=1)

    # Select unlabeled frames with high propagation confidence
    # that have bees (class 1 or 2)
    selected = []
    for i, stem in enumerate(stems):
        if y[i] == -1 and lgc_confidence[i] > 0.8 and y_lgc[i] > 0:
            selected.append(stem)

    with open(results_dir / "selected_frames.txt", "w") as f:
        for s in selected:
            f.write(f"{s}\n")

    print(f"\n  Selected {len(selected)} high-confidence frames for augmentation")

    # ── Save metrics ──
    metrics = {
        "label_pct": label_pct,
        "k": k,
        "alpha": alpha,
        "n_labeled": int(n_labeled),
        "n_unlabeled": int(n_unlabeled),
        "ghf_accuracy": round(ghf_acc, 4),
        "lgc_accuracy": round(lgc_acc, 4),
        "n_selected": len(selected)
    }

    with open(results_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)

    print(f"\nResults saved to {results_dir}/")

    # ── Classification report ──
    print(f"\n{'='*60}")
    print("GHF Classification Report (unlabeled frames):")
    print(classification_report(y_true[unlabeled_mask], y_ghf[unlabeled_mask],
                                target_names=["no_bees", "1-2_bees", "3+_bees"],
                                zero_division=0))

    print("LGC Classification Report (unlabeled frames):")
    print(classification_report(y_true[unlabeled_mask], y_lgc[unlabeled_mask],
                                target_names=["no_bees", "1-2_bees", "3+_bees"],
                                zero_division=0))

    return metrics


def _save_predictions(stems, predictions, path):
    """Save frame-level predictions to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "predicted_class"])
        for stem, pred in zip(stems, predictions):
            writer.writerow([stem, int(pred)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph-based label propagation")
    parser.add_argument("--splits_dir", type=str, default="splits")
    parser.add_argument("--label_pct", type=int, default=10, choices=[5, 10, 20])
    parser.add_argument("--k", type=int, default=10,
                        help="Number of neighbors for k-NN graph")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="LGC alpha parameter (0=clamp labels, 1=ignore labels)")
    parser.add_argument("--model", type=str, default="best.pt",
                        help="Model for feature extraction")
    args = parser.parse_args()

    run_label_propagation(
        splits_dir=args.splits_dir,
        label_pct=args.label_pct,
        k=args.k,
        alpha=args.alpha,
        model_path=args.model
    )
