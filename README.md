# How to Run — Semi-Supervised Bee Detection

## Prerequisites

```bash
pip install ultralytics opencv-python numpy pandas matplotlib scikit-learn openpyxl
```

## File Setup

Put all files in one folder:

```
project/
├── best.pt                    # trained YOLO model
├── 2024-10-25_1848.mp4        # bee video
├── tracker.py                 # detection + tracking pipeline
├── live_tracker.py            # real-time tracking + Supabase
├── data_split.py              # create label splits
├── pseudo_labeling.py         # self-training experiments
├── label_propagation.py       # GHF + LGC experiments
└── evaluate.py                # generate comparison plots
```

---

## 1. Run Bee Tracker (detection + tracking)
run:

```bash
python tracker.py
```

**Output** (in `output/` folder):
- `tracked_video.mp4` — annotated video with bounding boxes, IDs, trails
- `bee_telemetry.csv` — per-frame positions and speeds
- `feeder_visits.csv` — feeder zone entry/exit log
- `trophallaxis_events.csv` — food exchange events
- `bee_summary_statistics.xlsx` — per-bee stats
- `speed_plot.png` — speed over time graph

---

## 2. Run Live Tracker (real-time + Supabase)

Create a `.env` file with your Supabase credentials:

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
```

Then run:

```bash
# From camera
python live_tracker.py

# From video file
python live_tracker.py --video 2024-10-25_1848.mp4

# Specify camera index
python live_tracker.py --camera 1
```

Press `Esc` to stop.

---



## 3. Run Semi-Supervised Experiments

### Step 1: Generate dataset splits

```bash
python data_split.py --from_video 2024-10-25_1848.mp4 --model best.pt --output_dir splits
```

Creates 5%, 10%, 20% labeled subsets + test set from the video. Takes ~2 minutes.

### Step 2: Run pseudo-labeling (self-training)

```bash
python pseudo_labeling.py --splits_dir splits --label_pct 5 --tau 0.8 --rounds 3 --epochs 25 --model best.pt
```

Optional — run at other label budgets:

```bash
python pseudo_labeling.py --splits_dir splits --label_pct 10 --tau 0.8 --rounds 3 --epochs 25 --model best.pt
python pseudo_labeling.py --splits_dir splits --label_pct 20 --tau 0.8 --rounds 3 --epochs 25 --model best.pt
```

Each run takes ~15-20 min on GPU, ~30 min on CPU.

### Step 3: Run label propagation (GHF + LGC)

```bash
python label_propagation.py --splits_dir splits --label_pct 5 --k 10 --alpha 0.2 --model best.pt
```

Optional — run at other label budgets:

```bash
python label_propagation.py --splits_dir splits --label_pct 10 --k 10 --alpha 0.2 --model best.pt
python label_propagation.py --splits_dir splits --label_pct 20 --k 10 --alpha 0.2 --model best.pt
```

Each run takes ~2 minutes (no training, just graph computation).

### Step 4: Generate comparison plots

```bash
python evaluate.py --splits_dir splits --epochs 25 --model best.pt
```

**Output** (in `results/` folder):
- `comparison_table.csv` — all methods compared
- `learning_curve.png` — mAP vs label budget
- `bar_chart.png` — grouped bar chart

---

## Quick Reference

| Script | What it does | Time |
|--------|-------------|------|
| `tracker.py` | Offline detection + tracking + telemetry | ~5 min |
| `live_tracker.py` | Real-time tracking + Supabase upload | continuous |
| `data_split.py` | Create label budget splits | ~2 min |
| `pseudo_labeling.py` | Self-training loop | ~15-30 min |
| `label_propagation.py` | GHF + LGC graph propagation | ~2 min |
| `evaluate.py` | Compare methods + generate plots | ~15-30 min |
