"""
evaluate.py — Compare All Semi-Supervised Methods
====================================================
Runs evaluation across all label budgets and methods, generating:
  - Comparison table (mAP, precision, recall per method per budget)
  - Learning curve plot (mAP vs. label budget)
  - Per-method performance bar chart

Usage:
    python evaluate.py --splits_dir splits/ --model best.pt

    # Or evaluate a specific trained model:
    python evaluate.py --eval_model results/pseudo_labeling_10pct/round_3/train/weights/best.pt

Outputs:
    results/
        comparison_table.csv
        learning_curve.png
        bar_chart.png
"""

import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def evaluate_model(model_path, data_yaml, split="test"):
    """Evaluate a YOLO model and return metrics dict."""
    model = YOLO(model_path)
    results = model.val(data=data_yaml, split=split, verbose=False)

    return {
        "mAP50": round(float(results.box.map50), 4),
        "mAP50_95": round(float(results.box.map), 4),
        "precision": round(float(results.box.mp), 4),
        "recall": round(float(results.box.mr), 4),
    }


def train_baseline(splits_dir, label_pct, epochs=50, base_model="yolo11n.pt"):
    """Train a supervised-only baseline on a given label budget."""
    splits_dir = Path(splits_dir)
    results_dir = Path(f"results/supervised_{label_pct}pct")
    results_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = splits_dir / f"labeled_{label_pct}pct.yaml"

    print(f"\nTraining supervised baseline ({label_pct}% labels)...")
    model = YOLO(base_model)
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=480,
        batch=32,
        project=str(results_dir),
        name="train",
        exist_ok=True,
        verbose=False,
        degrees=180,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        scale=0.3,
        patience=50,
    )

    best_path = results_dir / "train" / "weights" / "best.pt"
    if not best_path.exists():
        best_path = results_dir / "train" / "weights" / "last.pt"

    metrics = evaluate_model(str(best_path), str(yaml_path))
    metrics["method"] = "supervised"
    metrics["label_pct"] = label_pct
    metrics["model_path"] = str(best_path)

    print(f"  Supervised {label_pct}%: mAP@0.5 = {metrics['mAP50']}")
    return metrics


def collect_pseudo_labeling_results(label_pct):
    """Collect the best round results from pseudo-labeling experiments."""
    results_dirs = list(Path("results").glob(f"pseudo_labeling_{label_pct}pct_*"))

    best_metrics = None
    for rd in results_dirs:
        metrics_path = rd / "metrics.csv"
        if not metrics_path.exists():
            continue

        with open(metrics_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            continue

        # Get the best round by mAP50
        best_row = max(rows, key=lambda r: float(r["mAP50"]))
        m = {
            "method": "pseudo_labeling",
            "label_pct": label_pct,
            "mAP50": float(best_row["mAP50"]),
            "mAP50_95": float(best_row.get("mAP50_95", 0)),
            "precision": float(best_row["precision"]),
            "recall": float(best_row["recall"]),
            "best_round": int(best_row["round"]),
            "n_train_frames": int(best_row["n_train_frames"]),
        }

        if best_metrics is None or m["mAP50"] > best_metrics["mAP50"]:
            best_metrics = m

    return best_metrics


def collect_label_propagation_results(label_pct):
    """Collect label propagation results."""
    rd = Path(f"results/label_propagation_{label_pct}pct")
    metrics_path = rd / "metrics.csv"

    if not metrics_path.exists():
        return None

    with open(metrics_path) as f:
        reader = csv.DictReader(f)
        row = next(reader)

    return {
        "method": "label_propagation",
        "label_pct": label_pct,
        "ghf_accuracy": float(row["ghf_accuracy"]),
        "lgc_accuracy": float(row["lgc_accuracy"]),
        "n_selected": int(row["n_selected"]),
    }


def run_full_evaluation(splits_dir, epochs=50, base_model="yolo11n.pt"):
    """Run or collect all experiments and generate comparison plots."""
    splits_dir = Path(splits_dir)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    all_results = []

    # ── Supervised baselines ──
    for pct in [5, 10, 20, 100]:
        # Check if already trained
        existing = results_dir / f"supervised_{pct}pct" / "train" / "weights" / "best.pt"
        if existing.exists():
            yaml_path = splits_dir / f"labeled_{pct}pct.yaml"
            metrics = evaluate_model(str(existing), str(yaml_path))
            metrics["method"] = "supervised"
            metrics["label_pct"] = pct
            print(f"  Supervised {pct}%: mAP@0.5 = {metrics['mAP50']} (cached)")
        else:
            metrics = train_baseline(splits_dir, pct, epochs, base_model)

        all_results.append(metrics)

    # ── Pseudo-labeling results ──
    for pct in [5, 10, 20]:
        pl_metrics = collect_pseudo_labeling_results(pct)
        if pl_metrics:
            all_results.append(pl_metrics)
            print(f"  Pseudo-labeling {pct}%: mAP@0.5 = {pl_metrics['mAP50']}")

    # ── Label propagation results ──
    for pct in [5, 10, 20]:
        lp_metrics = collect_label_propagation_results(pct)
        if lp_metrics:
            all_results.append(lp_metrics)
            print(f"  Label propagation {pct}%: GHF acc = {lp_metrics['ghf_accuracy']}, "
                  f"LGC acc = {lp_metrics['lgc_accuracy']}")

    # ── Save comparison table ──
    csv_path = results_dir / "comparison_table.csv"
    if all_results:
        keys = set()
        for r in all_results:
            keys.update(r.keys())
        keys = sorted(keys)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)

        print(f"\nComparison table saved to {csv_path}")

    # ── Generate plots ──
    _plot_learning_curve(all_results, results_dir)
    _plot_bar_chart(all_results, results_dir)

    return all_results


def _plot_learning_curve(results, output_dir):
    """Plot mAP@0.5 vs label budget for each method."""
    plt.figure(figsize=(10, 6))

    methods = {
        "supervised": {"color": "#E55934", "marker": "o", "label": "Supervised Only"},
        "pseudo_labeling": {"color": "#E2A832", "marker": "s", "label": "Pseudo-Labeling"},
    }

    for method, style in methods.items():
        method_results = [r for r in results if r.get("method") == method and "mAP50" in r]
        if not method_results:
            continue

        pcts = [r["label_pct"] for r in method_results]
        maps = [r["mAP50"] for r in method_results]

        # Sort by label percentage
        paired = sorted(zip(pcts, maps))
        pcts, maps = zip(*paired)

        plt.plot(pcts, maps, color=style["color"], marker=style["marker"],
                 linewidth=2, markersize=8, label=style["label"])

    plt.xlabel("Label Budget (%)", fontsize=13)
    plt.ylabel("mAP@0.5", fontsize=13)
    plt.title("Semi-Supervised Bee Detection: Performance vs. Label Budget", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks([5, 10, 20, 100])
    plt.ylim(0, 1.0)
    plt.tight_layout()

    path = output_dir / "learning_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Learning curve saved to {path}")


def _plot_bar_chart(results, output_dir):
    """Plot grouped bar chart comparing methods at each label budget."""
    fig, ax = plt.subplots(figsize=(10, 6))

    budgets = [5, 10, 20]
    methods_order = ["supervised", "pseudo_labeling"]
    colors = {"supervised": "#E55934", "pseudo_labeling": "#E2A832"}
    labels = {"supervised": "Supervised Only", "pseudo_labeling": "Pseudo-Labeling"}

    x = np.arange(len(budgets))
    width = 0.3
    n_methods = len(methods_order)

    for i, method in enumerate(methods_order):
        method_results = {r["label_pct"]: r["mAP50"]
                          for r in results
                          if r.get("method") == method and "mAP50" in r}
        vals = [method_results.get(b, 0) for b in budgets]
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=labels[method],
                      color=colors[method], edgecolor="white", linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                        f"{val:.2f}", ha='center', va='bottom', fontsize=9)

    # Add 100% baseline line
    full_sup = [r for r in results if r.get("method") == "supervised"
                and r.get("label_pct") == 100 and "mAP50" in r]
    if full_sup:
        ax.axhline(y=full_sup[0]["mAP50"], color="gray", linestyle="--",
                   linewidth=1, label=f"100% Supervised ({full_sup[0]['mAP50']:.2f})")

    ax.set_xlabel("Label Budget (%)", fontsize=13)
    ax.set_ylabel("mAP@0.5", fontsize=13)
    ax.set_title("Method Comparison by Label Budget", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}%" for b in budgets])
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()

    path = output_dir / "bar_chart.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Bar chart saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all methods")
    parser.add_argument("--splits_dir", type=str, default="splits")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="Base model for training baselines")
    parser.add_argument("--eval_model", type=str, default=None,
                        help="Evaluate a specific model only")
    parser.add_argument("--data_yaml", type=str, default=None,
                        help="Data YAML for single model evaluation")
    args = parser.parse_args()

    if args.eval_model and args.data_yaml:
        metrics = evaluate_model(args.eval_model, args.data_yaml)
        print(f"Model: {args.eval_model}")
        print(f"  mAP@0.5:   {metrics['mAP50']}")
        print(f"  mAP@50-95: {metrics['mAP50_95']}")
        print(f"  Precision: {metrics['precision']}")
        print(f"  Recall:    {metrics['recall']}")
    else:
        run_full_evaluation(args.splits_dir, args.epochs, args.model)
