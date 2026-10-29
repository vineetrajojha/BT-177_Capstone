# BEVFusionLite — Agent Rules & Engineering Constraints

## 1. Golden Rule: Prompt Path Fidelity
- Strictly follow the prompt path: **Prompt 0 through Prompt 12** in order.
- Do NOT skip steps or jump ahead to custom architectures. Prompts 3 (PointPillars Reference Baseline), 4 (Literature Review & Novelty), and 5 (Benchmark Tool) are mandatory prerequisites before Prompt 6.

## 2. Model Architecture Specifications
- **Heavy Baseline**: ResNet-50 camera backbone, 128 image channels, 64 LiDAR channels.
- **Light Model**: ResNet-18 camera backbone, 64 image channels, 32 LiDAR channels.
- Both models must be served by a unified model class configurable via YAML.

## 3. Data Integrity & Verification
- Only use real, official datasets (KITTI, nuScenes). Never use synthetic data without explicit marking.
- Any calibration or preprocessing must be verified against ground truth labels and mathematically documented.
- No image distortion or arbitrary resizing that breaks perspective geometry: use pad-only image handling (`scale = 1.0`, unpadded width $W_{\text{orig}}$ for horizontal flips).

## 4. Compute & Environment Constraints
- Local development is on Windows / RTX 3050 (or CPU).
- The local disk has limited space (~10 GB free on `C:`).
- **Full training (80 epochs, 3 seeds) MUST run on Kaggle or Google Colab**, not locally.
- Keep all unit tests passing (`pytest tests/ -v`).
