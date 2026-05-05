"""
pseudo_labeling.py — Self-Training Loop for Semi-Supervised Bee Detection
==========================================================================
Implements the pseudo-labeling (self-training) algorithm:
  1. Train YOLO11n on small labeled subset
  2. Run inference on unlabeled frames
  3. Keep predictions above confidence threshold τ as pseudo-labels
  4. Add pseudo-labeled frames to training set
  5. Retrain and repeat

Usage:
    python pseudo_labeling.py --splits_dir splits/ --label_pct 10 --tau 0.8 --rounds 3

Outputs:
    results/pseudo_labeling/
        round_0/              — baseline model (trained on labeled subset only)
        round_1/              — after 1st self-training iteration
        round_2/              — after 2nd iteration
        round_3/              — after 3rd iteration
        metrics.csv           — mAP/precision/recall per round
"""

import os
import shutil
import csv
import argparse
from pathlib import Path
from ultralytics import YOLO


def run_pseudo_labeling(splits_dir, label_pct, tau, rounds, epochs, model_base):
    """
    Execute the pseudo-labeling self-training loop.

    Args:
        splits_dir: Directory with labeled/unlabeled splits from data_split.py
        label_pct: Which label budget to use (5, 10, or 20)
        tau: Confidence threshold for pseudo-labels
        rounds: Number of self-training iterations
        epochs: Training epochs per round
        model_base: Base YOLO model to start from (e.g., yolo11n.pt)
    """
    splits_dir = Path(splits_dir)
    results_dir = Path(f"results/pseudo_labeling_{label_pct}pct_tau{tau}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Paths
    labeled_dir = splits_dir / f"labeled_{label_pct}pct"
    unlabeled_dir = splits_dir / f"unlabeled_{label_pct}pct"
    test_dir = splits_dir / "test"
    yaml_path = splits_dir / f"labeled_{label_pct}pct.yaml"

    # Working directory for current training set (starts as labeled subset)
    work_dir = results_dir / "working_set"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    shutil.copytree(labeled_dir, work_dir)

    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    n_labeled_initial = len([f for f in (work_dir / "images").iterdir() if f.suffix.lower() in IMG_EXTS])
    n_unlabeled = len([f for f in (unlabeled_dir / "images").iterdir() if f.suffix.lower() in IMG_EXTS])
    print(f"Starting pseudo-labeling: {label_pct}% labeled ({n_labeled_initial} frames)")
    print(f"Unlabeled pool: {n_unlabeled} frames")
    print(f"Confidence threshold τ = {tau}, Rounds = {rounds}\n")

    metrics_log = []
    current_model_path = model_base

    for round_num in range(rounds + 1):
        round_dir = results_dir / f"round_{round_num}"
        round_dir.mkdir(exist_ok=True)

        # ── Train ──
        print(f"{'='*60}")
        print(f"Round {round_num}: Training on {len([f for f in (work_dir / 'images').iterdir() if f.suffix.lower() in IMG_EXTS])} frames")
        print(f"{'='*60}")

        # Create YAML for current working set
        work_yaml = _create_work_yaml(work_dir, test_dir, round_dir / "data.yaml")

        model = YOLO(current_model_path)
        train_results = model.train(
            data=str(work_yaml),
            epochs=epochs,
            imgsz=480,
            batch=32,
            project=str(round_dir),
            name="train",
            exist_ok=True,
            verbose=False,
            # Augmentations matching your train.py
            degrees=180,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            scale=0.3,
            patience=50,
        )

        # Get best model from this round
        # Ultralytics may save to the project dir OR to runs/detect/
        # Check the actual save directory from train_results
        save_dir = Path(train_results.save_dir) if hasattr(train_results, 'save_dir') else round_dir / "train"
        best_model_path = save_dir / "weights" / "best.pt"
        if not best_model_path.exists():
            best_model_path = save_dir / "weights" / "last.pt"
        if not best_model_path.exists():
            # Fallback: search for the weights
            import glob
            candidates = glob.glob(str(round_dir / "**" / "weights" / "best.pt"), recursive=True)
            if not candidates:
                candidates = glob.glob("runs/**/weights/best.pt", recursive=True)
            if candidates:
                best_model_path = Path(sorted(candidates)[-1])  # most recent
            else:
                raise FileNotFoundError(f"Cannot find trained weights in {round_dir} or runs/")
        current_model_path = str(best_model_path)
        print(f"  Model saved to: {current_model_path}")

        # ── Evaluate on test set ──
        model = YOLO(current_model_path)
        val_results = model.val(data=str(work_yaml), split="test", verbose=False)

        metrics = {
            "round": round_num,
            "n_train_frames": len([f for f in (work_dir / "images").iterdir() if f.suffix in (".png",".jpg")]),
            "mAP50": round(float(val_results.box.map50), 4),
            "mAP50_95": round(float(val_results.box.map), 4),
            "precision": round(float(val_results.box.mp), 4),
            "recall": round(float(val_results.box.mr), 4),
        }
        metrics_log.append(metrics)

        print(f"\n  Round {round_num} Results:")
        print(f"    mAP@0.5:    {metrics['mAP50']}")
        print(f"    Precision:  {metrics['precision']}")
        print(f"    Recall:     {metrics['recall']}")
        print(f"    Train size: {metrics['n_train_frames']} frames\n")

        # ── Generate pseudo-labels (skip on last round) ──
        if round_num < rounds:
            n_added = _generate_pseudo_labels(
                model_path=current_model_path,
                unlabeled_dir=unlabeled_dir,
                work_dir=work_dir,
                tau=tau
            )
            print(f"  Added {n_added} pseudo-labeled frames (τ ≥ {tau})\n")

            if n_added == 0:
                print("  No new pseudo-labels above threshold. Stopping early.")
                break

    # ── Save metrics ──
    metrics_csv = results_dir / "metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_log[0].keys())
        writer.writeheader()
        writer.writerows(metrics_log)

    print(f"\nMetrics saved to {metrics_csv}")
    print(f"Best model: {current_model_path}")

    return metrics_log


def _generate_pseudo_labels(model_path, unlabeled_dir, work_dir, tau):
    """
    Run inference on unlabeled frames and add high-confidence predictions
    as pseudo-labels to the working training set.
    """
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    model = YOLO(model_path)
    unlabeled_imgs = sorted([f for f in (unlabeled_dir / "images").iterdir()
                             if f.suffix.lower() in IMG_EXTS])
    n_added = 0

    work_img_dir = work_dir / "images"
    work_lbl_dir = work_dir / "labels"
    work_lbl_dir.mkdir(exist_ok=True)

    already_added = set(f.stem for f in work_img_dir.iterdir() if f.suffix.lower() in IMG_EXTS)

    # First pass: collect all confidence scores to report distribution
    all_confs = []
    for img_path in unlabeled_imgs[:50]:  # sample first 50 for diagnostics
        results = model(str(img_path), verbose=False)
        if len(results[0].boxes) > 0:
            all_confs.extend(results[0].boxes.conf.cpu().numpy().tolist())

    if all_confs:
        import statistics
        print(f"  Confidence distribution (sample): "
              f"min={min(all_confs):.3f}, median={statistics.median(all_confs):.3f}, "
              f"max={max(all_confs):.3f}, mean={statistics.mean(all_confs):.3f}")
        print(f"  Frames above τ={tau}: {sum(1 for c in all_confs if c >= tau)}/{len(all_confs)} detections")

        # Auto-adjust threshold if nothing passes
        if all(c < tau for c in all_confs):
            adjusted_tau = round(statistics.median(all_confs) * 0.9, 2)
            print(f"  WARNING: No detections above τ={tau}. Auto-adjusting to τ={adjusted_tau}")
            tau = adjusted_tau
    else:
        print("  WARNING: No detections found in unlabeled frames")
        return 0

    # Second pass: generate pseudo-labels
    for img_path in unlabeled_imgs:
        if img_path.stem in already_added:
            continue

        results = model(str(img_path), verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            continue

        confs = boxes.conf.cpu().numpy()

        # Keep only boxes above threshold
        high_conf_mask = confs >= tau
        if high_conf_mask.sum() == 0:
            continue

        # Write pseudo-labels in YOLO format
        img = results[0].orig_img
        h, w = img.shape[:2]
        xyxy = boxes.xyxy.cpu().numpy()

        lbl_lines = []
        for i in range(len(xyxy)):
            if not high_conf_mask[i]:
                continue
            x1, y1, x2, y2 = xyxy[i]
            cx = ((x1 + x2) / 2.0) / w
            cy = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lbl_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if lbl_lines:
            # Copy image to working set
            shutil.copy2(img_path, work_img_dir / img_path.name)
            # Write pseudo-label
            with open(work_lbl_dir / f"{img_path.stem}.txt", "w") as f:
                f.write("\n".join(lbl_lines) + "\n")
            n_added += 1

    return n_added


def _create_work_yaml(work_dir, test_dir, yaml_path):
    """Create a YOLO data.yaml pointing to the current working set."""
    train_path = str((work_dir / "images").resolve())
    test_path = str((test_dir / "images").resolve())

    with open(yaml_path, "w") as f:
        f.write(f"train: {train_path}\n")
        f.write(f"val: {test_path}\n")
        f.write(f"test: {test_path}\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['bee']\n")

    return yaml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo-labeling self-training")
    parser.add_argument("--splits_dir", type=str, default="splits")
    parser.add_argument("--label_pct", type=int, default=10, choices=[5, 10, 20])
    parser.add_argument("--tau", type=float, default=0.8,
                        help="Confidence threshold for pseudo-labels")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Number of self-training iterations")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs per round")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="Base model to start from (yolo11n.pt or yolo11s.pt)")
    args = parser.parse_args()

    metrics = run_pseudo_labeling(
        splits_dir=args.splits_dir,
        label_pct=args.label_pct,
        tau=args.tau,
        rounds=args.rounds,
        epochs=args.epochs,
        model_base=args.model
    )

    print("\n=== Summary ===")
    for m in metrics:
        print(f"  Round {m['round']}: mAP@0.5 = {m['mAP50']}, "
              f"P = {m['precision']}, R = {m['recall']}, "
              f"Frames = {m['n_train_frames']}")
