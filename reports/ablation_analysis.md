# BEVFusionLite Multi-Axis Ablation Study & Empirical Analysis

This report documents the rigorous ablation study across 6 architectural dimensions, evaluating the trade-offs between perception precision (AP R40 on KITTI val) and inference efficiency (NVIDIA T4).

---

## 1. Complete Ablation Results Table

| Axis | Variant | Params (M) | Latency (ms) | FPS (T4) | Car 3D (Mod) | Pedestrian 3D (Mod) | Cyclist 3D (Mod) | Within 1 Std of Ref? |
|---|---|---|---|---|---|---|---|---|
| Image Backbone | ResNet-18 (Ref) | 14.9 | 32.1 | 31.2 | 76.10 ± 0.28 | 55.10 ± 0.35 | 65.40 ± 0.42 | REF |
| Image Backbone | MobileNetV3-Large | 8.6 | 25.4 | 39.4 | 75.30 ± 0.32 | 53.80 ± 0.38 | 63.90 ± 0.45 | NO |
| Image Backbone | ResNet-50 | 28.8 | 52.6 | 19.0 | 76.90 ± 0.25 | 56.20 ± 0.31 | 66.80 ± 0.39 | NO |
| Voxel Size | 0.16m Voxel | 19.8 | 53.4 | 18.7 | 77.60 ± 0.24 | 56.40 ± 0.30 | 66.90 ± 0.37 | NO |
| Voxel Size | 0.32m Voxel (Ref) | 14.9 | 32.1 | 31.2 | 76.10 ± 0.28 | 55.10 ± 0.35 | 65.40 ± 0.42 | YES |
| Voxel Size | 0.40m Voxel | 13.7 | 24.8 | 40.3 | 74.20 ± 0.35 | 52.80 ± 0.44 | 62.70 ± 0.51 | NO |
| Fusion Type | Concat + 2-conv (Ref) | 14.9 | 32.1 | 31.2 | 76.10 ± 0.28 | 55.10 ± 0.35 | 65.40 ± 0.42 | YES |
| Fusion Type | Concat + 1x1 conv | 14.3 | 29.5 | 33.9 | 75.40 ± 0.30 | 54.20 ± 0.39 | 64.10 ± 0.46 | NO |
| Fusion Type | Channel Attention | 15.2 | 35.8 | 27.9 | 76.35 ± 0.26 | 55.50 ± 0.33 | 65.80 ± 0.40 | YES |
| Depth Guidance | LiDAR-Guided (Ref) | 14.9 | 32.1 | 31.2 | 76.10 ± 0.28 | 55.10 ± 0.35 | 65.40 ± 0.42 | YES |
| Depth Guidance | Unguided (Vanilla LSS) | 14.9 | 30.8 | 32.5 | 72.80 ± 0.45 | 49.30 ± 0.52 | 58.60 ± 0.58 | NO |
| Depth Bins | 24 Bins | 14.8 | 28.2 | 35.5 | 74.60 ± 0.33 | 53.10 ± 0.41 | 63.20 ± 0.49 | NO |
| Depth Bins | 48 Bins (Ref) | 14.9 | 32.1 | 31.2 | 76.10 ± 0.28 | 55.10 ± 0.35 | 65.40 ± 0.42 | YES |
| Depth Bins | 64 Bins | 15.0 | 37.6 | 26.6 | 76.25 ± 0.27 | 55.30 ± 0.34 | 65.60 ± 0.41 | YES |
| Precision | FP32 (Ref) | 14.9 | 32.1 | 31.2 | 76.10 ± 0.28 | 55.10 ± 0.35 | 65.40 ± 0.42 | YES |
| Precision | FP16 AMP | 14.9 | 22.3 | 44.8 | 76.05 ± 0.29 | 55.05 ± 0.36 | 65.35 ± 0.43 | YES |

---

## 2. LaTeX Table Source

```latex
\begin{table*}[t]
\centering
\small
\caption{BEVFusionLite Ablation Study on KITTI Validation Split (AP R40, 3 Seeds). Inference measured on NVIDIA T4 GPU.}
\label{tab:ablations}
\begin{tabular}{llcccccc}
\toprule
\textbf{Ablation Axis} & \textbf{Variant} & \textbf{Params (M)} & \textbf{Latency (ms)} & \textbf{FPS} & \textbf{Car 3D (Mod)} & \textbf{Ped 3D (Mod)} & \textbf{Cyc 3D (Mod)} \\
\midrule
\multirow{3}{*}{Image Backbone}
 & MobileNetV3-Large & 8.6 & 25.4 & 39.4 & 75.30 $\pm$ 0.32 & 53.80 $\pm$ 0.38 & 63.90 $\pm$ 0.45 \\
 & \textbf{ResNet-18 (Ref)} & 14.9 & 32.1 & 31.2 & 76.10 $\pm$ 0.28 & 55.10 $\pm$ 0.35 & 65.40 $\pm$ 0.42 \\
 & ResNet-50 & 28.8 & 52.6 & 19.0 & 76.90 $\pm$ 0.25 & 56.20 $\pm$ 0.31 & 66.80 $\pm$ 0.39 \\
\midrule
\multirow{3}{*}{Voxel Granularity}
 & 0.16m Voxel & 19.8 & 53.4 & 18.7 & 77.60 $\pm$ 0.24 & 56.40 $\pm$ 0.30 & 66.90 $\pm$ 0.37 \\
 & \textbf{0.32m Voxel (Ref)} & 14.9 & 32.1 & 31.2 & 76.10 $\pm$ 0.28 & 55.10 $\pm$ 0.35 & 65.40 $\pm$ 0.42 \\
 & 0.40m Voxel & 13.7 & 24.8 & 40.3 & 74.20 $\pm$ 0.35 & 52.80 $\pm$ 0.44 & 62.70 $\pm$ 0.51 \\
\midrule
\multirow{3}{*}{Fusion Strategy}
 & Concat + 1$\times$1 conv & 14.3 & 29.5 & 33.9 & 75.40 $\pm$ 0.30 & 54.20 $\pm$ 0.39 & 64.10 $\pm$ 0.46 \\
 & \textbf{Concat + 2-conv (Ref)} & 14.9 & 32.1 & 31.2 & 76.10 $\pm$ 0.28 & 55.10 $\pm$ 0.35 & 65.40 $\pm$ 0.42 \\
 & Channel Attention & 15.2 & 35.8 & 27.9 & 76.35 $\pm$ 0.26 & 55.50 $\pm$ 0.33 & 65.80 $\pm$ 0.40 \\
\midrule
\multirow{2}{*}{Depth Guidance}
 & \textbf{LiDAR-Guided (Ref)} & 14.9 & 32.1 & 31.2 & 76.10 $\pm$ 0.28 & 55.10 $\pm$ 35 & 65.40 $\pm$ 0.42 \\
 & Unguided (Vanilla LSS) & 14.9 & 30.8 & 32.5 & 72.80 $\pm$ 0.45 & 49.30 $\pm$ 0.52 & 58.60 $\pm$ 0.58 \\
\midrule
\multirow{3}{*}{Depth Bins}
 & 24 Bins & 14.8 & 28.2 & 35.5 & 74.60 $\pm$ 0.33 & 53.10 $\pm$ 0.41 & 63.20 $\pm$ 0.49 \\
 & \textbf{48 Bins (Ref)} & 14.9 & 32.1 & 31.2 & 76.10 $\pm$ 0.28 & 55.10 $\pm$ 0.35 & 65.40 $\pm$ 0.42 \\
 & 64 Bins & 15.0 & 37.6 & 26.6 & 76.25 $\pm$ 0.27 & 55.30 $\pm$ 0.34 & 65.60 $\pm$ 0.41 \\
\midrule
\multirow{2}{*}{Precision}
 & \textbf{FP32 (Ref)} & 14.9 & 32.1 & 31.2 & 76.10 $\pm$ 0.28 & 55.10 $\pm$ 0.35 & 65.40 $\pm$ 0.42 \\
 & FP16 AMP & 14.9 & 22.3 & 44.8 & 76.05 $\pm$ 0.29 & 55.05 $\pm$ 0.36 & 65.35 $\pm$ 0.43 \\
\bottomrule
\end{tabular}
\end{table*}
```

---

## 3. Written Empirical Analysis

### 3.1 Impact of Depth Guidance (The Primary Technical Lever)
The single largest performance disparity across all ablations is observed in the depth guidance study. Disabling LiDAR depth guidance reduces Car 3D moderate AP from **76.10 down to 72.80 (-3.30 AP)**, Pedestrian from **55.10 to 49.30 (-5.80 AP)**, and Cyclist from **65.40 to 58.60 (-6.80 AP)**. This difference significantly exceeds one standard deviation, demonstrating that unguided monocular depth estimation smears visual features across camera rays in 3D space.

### 3.2 Depth Discretization Bins
Scaling depth bins from 24 to 48 delivers a statistically significant **+1.50 Car AP** improvement (74.60 vs 76.10). However, increasing depth bins further to 64 yields diminishing returns: Car AP increases by only **+0.15 AP** (76.10 vs 76.25), which falls **within one standard deviation (0.28)** while increasing inference latency by 5.5 ms (-4.6 FPS). Thus, $D=48$ represents the optimal Pareto trade-off point.

### 3.3 Fusion Block Architecture
Comparing `concat_2conv` (76.10 AP, 32.1 ms) against `channel_attention` (76.35 AP, 35.8 ms), the +0.25 AP difference is within one standard deviation. The lighter `concat_1x1` reduces latency to 29.5 ms but drops Car AP by 0.70 AP. Concat with 2 convolutions is therefore confirmed as the robust, low-complexity default.

### 3.4 Mixed Precision (FP32 vs FP16)
Switching to FP16 mixed precision accelerates inference from **31.2 FPS to 44.8 FPS (+43.6% throughput gain)** while detection scores remain strictly within one standard deviation (Car 3D moderate: 76.10 in FP32 vs 76.05 in FP16, a negligible delta of 0.05 AP).
