# BuzzSpot Pollinator Detection

Object-detection experiments developed for the BuzzSpot Challenge, progressing from YOLO11 and YOLO26 baselines to RF-DETR Large, higher-resolution training, cross-model ensembling, and causal temporal inference.

**Best result:** **0.405 mAP@0.5:0.95** on the hidden competition test set, placing **10th out of 36 participants**.

## Repository contents

```text
.
├── notebooks/                  # 15 experiments in chronological order
├── results/
│   ├── experiment_summary.csv  # Configurations, metrics, and outcomes
│   ├── best_model_metrics.csv  # Hidden-test breakdown for the best model
│   └── notebook_manifest.csv    # Original filenames and SHA-256 hashes
├── docs/
│   └── BuzzSpot_Challenge_Final_Report.pdf
├── scripts/
│   ├── build_experiment_summary.py
│   └── verify_notebooks.py
├── requirements.txt
└── .gitignore
```

The notebooks are copied byte-for-byte from the uploaded files; only their filenames and repository locations were changed. `results/notebook_manifest.csv` records the original filename and SHA-256 hash for each notebook.

## Experimental progression

| # | Notebook | Purpose | Local mAP@0.5:0.95 | Hidden-test mAP@0.5:0.95 |
|---:|---|---|---:|---:|
| 01 | [`01_yolo11_baseline.ipynb`](notebooks/01_yolo11_baseline.ipynb) | Initial clean training, validation, inference, and submission pipeline. | 0.3832 | — |
| 02 | [`02_yolo26s_baseline.ipynb`](notebooks/02_yolo26s_baseline.ipynb) | Longer barebones fine-tune to test whether additional training improved generalization. | 0.3928 | — |
| 03 | [`03_yolo26s_hoverfly_crops_24ep.ipynb`](notebooks/03_yolo26s_hoverfly_crops_24ep.ipynb) | Rare-class oversampling plus 3,506 hoverfly-centered crop images. | 0.3721 | — |
| 04 | [`04_yolo26s_tile_hardneg_training.ipynb`](notebooks/04_yolo26s_tile_hardneg_training.ipynb) | SAHI-style tiles, rare-class crops, and mined hard-negative clutter regions. | full frame 0.345; SAHI 0.173; combined 0.162 | — |
| 05 | [`05_yolo26m_trainval.ipynb`](notebooks/05_yolo26m_trainval.ipynb) | Full-frame train+validation final fit with rare-class image oversampling. | 0.837 | — |
| 06 | [`06_yolo26m_sliced700_finetune.ipynb`](notebooks/06_yolo26m_sliced700_finetune.ipynb) | Model B fine-tuned on 700×700 slices with 20% overlap, initialized from the full-frame model. | 0.797 | — |
| 07 | [`07_yolo26m_ab_ensemble.ipynb`](notebooks/07_yolo26m_ab_ensemble.ipynb) | Gated full-frame and sliced ensemble using the historically selected fusion configuration. | — | — |
| 08 | [`08_rfdetr_small_1120_8ep.ipynb`](notebooks/08_rfdetr_small_1120_8ep.ipynb) | Eight-epoch proof of concept at 1120 resolution with train+validation oversampling. | 0.7591 | — |
| 09 | [`09_rfdetr_large_1120_32ep.ipynb`](notebooks/09_rfdetr_large_1120_32ep.ipynb) | First serious 32-epoch RF-DETR Large run at 1120 resolution. | 0.8740 | 0.379 |
| 10 | [`10_size_routed_three_model_ensemble.ipynb`](notebooks/10_size_routed_three_model_ensemble.ipynb) | Hard-routed detections by COCO object-size bands, followed by class-aware NMS. | — | 0.369 |
| 11 | [`11_rfdetr_large_1344_32ep.ipynb`](notebooks/11_rfdetr_large_1344_32ep.ipynb) | Controlled resolution increase from 1120 to 1344 with the remaining setup frozen. | 0.8864 | 0.404906 |
| 12 | [`12_rfdetr_large_1344_temporal_inference.ipynb`](notebooks/12_rfdetr_large_1344_temporal_inference.ipynb) | Six-frame causal tracklets using Hungarian association, score fusion, class fusion, and proposal recovery. | baseline 0.8866; strongest temporal candidate 0.8863 | — |
| 13 | [`13_rfdetr_large_1536_fresh_32ep.ipynb`](notebooks/13_rfdetr_large_1536_fresh_32ep.ipynb) | Fresh controlled 1536-resolution run from the pretrained initialization. | 0.8892 | — |
| 14 | [`14_rfdetr_large_1344_to_1536_12ep.ipynb`](notebooks/14_rfdetr_large_1344_to_1536_12ep.ipynb) | Adaptation from the selected 1344 EMA checkpoint to 1536 for 12 epochs at half learning rate. | 0.9205 (approximately 0.921) | 0.395 |
| 15 | [`15_rfdetr_large_1344_to_1536_18ep_resume.ipynb`](notebooks/15_rfdetr_large_1344_to_1536_18ep_resume.ipynb) | True resume of the warm-started 1536 run from 12 to 18 total epochs. | 0.9275 | — |

### Validation caveat

After the annotated validation keyframes were added to training, later local validation results became **leaked** and are not honest estimates of generalization. They are included to document the runs, while hidden-test scores are reported only where a competition submission was evaluated.

## Best model

The strongest submission came from [`11_rfdetr_large_1344_32ep.ipynb`](notebooks/11_rfdetr_large_1344_32ep.ipynb):

- RF-DETR Large, 32 epochs, 1344 input resolution
- Official train + validation keyframes
- Rare-class image-level oversampling
- Batch size 2 with 8 gradient-accumulation steps
- EMA checkpoint selected for inference
- Confidence threshold 0.01, maximum 100 predictions per image

| Metric | AP |
|---|---:|
| Overall mAP@0.5:0.95 | 0.405 |
| Small / Medium / Large | 0.410 / 0.500 / 0.720 |
| Bee | 0.532 |
| Bumblebee | 0.435 |
| Hoverfly | 0.154 |
| Moth | 0.498 |

## Main findings

- Longer YOLO training and hoverfly-focused crops did not resolve the main generalization and class-discrimination problems.
- Inference-only slicing and the tile/hard-negative run increased false positives and underperformed the stronger full-frame models.
- RF-DETR Large generalized better than the YOLO configurations tested.
- Increasing RF-DETR input resolution from 1120 to 1344 improved hidden-test mAP from 0.379 to 0.405.
- Hard routing by predicted object size underperformed the strongest standalone model.
- Post-processing detections across six frames did not improve the single-frame 1344 baseline.
- Continued 1344→1536 fine-tuning raised leaked validation AP but reduced hidden-test performance.

## Running the notebooks

The notebooks were built for Google Colab and contain their own installation and Google Drive setup cells. They expect the BuzzSpot archives and trained checkpoints to be placed under the configured Drive paths. Update the path variables near the top of each notebook for a different environment.

The dataset, checkpoints, generated predictions, and submission archives are intentionally not included.

For a local environment, install a PyTorch build appropriate for your hardware, then install:

```bash
pip install -r requirements.txt
```

## Regenerating the experiment table

```bash
python scripts/build_experiment_summary.py
```

## Verifying notebook integrity

```bash
python scripts/verify_notebooks.py
```

## Push to GitHub

```bash
git init
git add .
git commit -m "Add BuzzSpot detection experiments"
git branch -M main
git remote add origin <your-repository-url>
git push -u origin main
```
