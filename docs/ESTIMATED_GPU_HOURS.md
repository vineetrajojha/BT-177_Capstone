# Estimated GPU Compute Hours & Execution Strategy

All latency, throughput, and memory numbers for official benchmarking target an **NVIDIA T4 GPU (16 GB VRAM)**.

## 1. Full Protocol (Theoretical Unconstrained Compute)

| Prompt / Phase | Model / Variant | Epochs | Seeds | Est. Time / Seed (T4) | Total Est. GPU Hours |
|---|---|---|---|---|---|
| **P3** | PointPillars Reference | 80 | 3 (0, 1, 2) | 11.5 h | ~34.5 h |
| **P6** | Heavy Baseline (ResNet-50 + 0.16m LiDAR) | 80 | 3 (0, 1, 2) | 17.0 h | ~51.0 h |
| **P7** | Light Model (ResNet-18 + 0.32m LiDAR) | 80 | 3 (0, 1, 2) | 8.5 h | ~25.5 h |
| **P7 Modalities** | Camera-Only & LiDAR-Only variants | 80 | 3 each | 7.0 h | ~42.0 h |
| **P8 Ablations** | 12 architectural variants | 80 | 3 each | 8.0 h | ~288.0 h |
| **P9** | nuScenes-mini (Heavy & Light) | 40 | 3 each | 6.0 h | ~36.0 h |
| **Total** | | | | | **~477.0 h** |

---

## 2. Recommended Efficient Protocol (for Kaggle / Colab Free T4 Tier)

Free Kaggle provides 30 GPU hours per week; Colab free tier provides ~4–8 hour continuous sessions. To deliver rigorous, publication-grade results within free-tier limits without compromising evaluation protocols:

| Prompt / Phase | Proposed Streamlined Protocol | Epochs | Seeds | Total Est. GPU Hours |
|---|---|---|---|---|
| **P3** | PointPillars Reference | 80 | Seed 0 (80 ep) + fast verification seeds | ~14 h |
| **P6** | Heavy Baseline | 80 | Seed 0 (80 ep) | ~17 h |
| **P7** | Light Model | 80 | Seed 0 (80 ep) | ~9 h |
| **P7 Modalities** | Camera-Only & LiDAR-Only | 40 | Seed 0 | ~7 h |
| **P8 Ablations** | 6 primary ablation axes (12 rows) | 20–40 | Seed 0 | ~36 h |
| **P9** | nuScenes-mini | 24 | Seed 0 | ~6 h |
| **Total** | | | | **~89 h** |

### Robust Session-Death Handling:
1. Every training run saves checkpoints every epoch: `checkpoint_epoch_<N>.pth`.
2. Automatic `--resume` capability detects the latest epoch and continues seamlessly.
3. Checkpoint auto-sync to Google Drive or Hugging Face repository on epoch completion.
