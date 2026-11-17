<!-- AUTO-GENERATED TABLES FROM RESULTS DIRECTORY. DO NOT HAND-EDIT. -->

### Table 1: Main Benchmark Comparison (KITTI Val Split)

| model                      | modality           | camera_backbone   |   voxel_size |   params_M |   gflops |   fps_t4 |   latency_ms_t4 |   peak_mem_mb |   car_3d_mod |   ped_3d_mod |   cyc_3d_mod |   car_bev_mod |   ped_bev_mod |   cyc_bev_mod |
|:---------------------------|:-------------------|:------------------|-------------:|-----------:|---------:|---------:|----------------:|--------------:|-------------:|-------------:|-------------:|--------------:|--------------:|--------------:|
| OpenPCDet_PointPillars_Ref | LiDAR              | nan               |         0.16 |        4.8 |     45.2 |     62   |            16.1 |          1150 |        77.28 |        52.29 |        62.68 |         88.61 |         58.74 |         65.5  |
| Heavy_Baseline_FP32        | Fusion (Cam+LiDAR) | ResNet-50         |         0.16 |       33.8 |    184.2 |     13.5 |            74.1 |          3820 |        78.45 |        56.8  |        67.2  |         89.9  |         62.4  |         70.1  |
| Light_Model_FP32           | Fusion (Cam+LiDAR) | ResNet-18         |         0.32 |       14.9 |     68.4 |     31.2 |            32.1 |          1610 |        76.1  |        55.1  |        65.4  |         88.1  |         60.8  |         68.2  |
| Light_Model_FP16           | Fusion (Cam+LiDAR) | ResNet-18         |         0.32 |       14.9 |     68.4 |     44.8 |            22.3 |          1120 |        76.05 |        55.05 |        65.35 |         88.05 |         60.75 |         68.15 |
| Light_LiDAR_Only_Ablation  | LiDAR              | nan               |         0.32 |        3.2 |     24.1 |     78   |            12.8 |           890 |        73.8  |        47.9  |        57.2  |         85.4  |         53.2  |         61.1  |
| Light_Camera_Only_Ablation | Camera             | ResNet-18         |         0.32 |       13.5 |     52.8 |     38.5 |            26   |          1420 |        19.4  |        12.8  |        14.2  |         26.1  |         17.5  |         19.3  |

### Table 2: Architectural Ablation Study (6 Axes, KITTI Val Split)

| axis           | variant                |   params_M |   latency_ms |   fps |   car_3d_mod |   car_3d_std |   ped_3d_mod |   ped_3d_std |   cyc_3d_mod |   cyc_3d_std |
|:---------------|:-----------------------|-----------:|-------------:|------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
| Image Backbone | ResNet-18 (Ref)        |       14.9 |         32.1 |  31.2 |        76.1  |         0.28 |        55.1  |         0.35 |        65.4  |         0.42 |
| Image Backbone | MobileNetV3-Large      |        8.6 |         25.4 |  39.4 |        75.3  |         0.32 |        53.8  |         0.38 |        63.9  |         0.45 |
| Image Backbone | ResNet-50              |       28.8 |         52.6 |  19   |        76.9  |         0.25 |        56.2  |         0.31 |        66.8  |         0.39 |
| Voxel Size     | 0.16m Voxel            |       19.8 |         53.4 |  18.7 |        77.6  |         0.24 |        56.4  |         0.3  |        66.9  |         0.37 |
| Voxel Size     | 0.32m Voxel (Ref)      |       14.9 |         32.1 |  31.2 |        76.1  |         0.28 |        55.1  |         0.35 |        65.4  |         0.42 |
| Voxel Size     | 0.40m Voxel            |       13.7 |         24.8 |  40.3 |        74.2  |         0.35 |        52.8  |         0.44 |        62.7  |         0.51 |
| Fusion Type    | Concat + 2-conv (Ref)  |       14.9 |         32.1 |  31.2 |        76.1  |         0.28 |        55.1  |         0.35 |        65.4  |         0.42 |
| Fusion Type    | Concat + 1x1 conv      |       14.3 |         29.5 |  33.9 |        75.4  |         0.3  |        54.2  |         0.39 |        64.1  |         0.46 |
| Fusion Type    | Channel Attention      |       15.2 |         35.8 |  27.9 |        76.35 |         0.26 |        55.5  |         0.33 |        65.8  |         0.4  |
| Depth Guidance | LiDAR-Guided (Ref)     |       14.9 |         32.1 |  31.2 |        76.1  |         0.28 |        55.1  |         0.35 |        65.4  |         0.42 |
| Depth Guidance | Unguided (Vanilla LSS) |       14.9 |         30.8 |  32.5 |        72.8  |         0.45 |        49.3  |         0.52 |        58.6  |         0.58 |
| Depth Bins     | 24 Bins                |       14.8 |         28.2 |  35.5 |        74.6  |         0.33 |        53.1  |         0.41 |        63.2  |         0.49 |
| Depth Bins     | 48 Bins (Ref)          |       14.9 |         32.1 |  31.2 |        76.1  |         0.28 |        55.1  |         0.35 |        65.4  |         0.42 |
| Depth Bins     | 64 Bins                |       15   |         37.6 |  26.6 |        76.25 |         0.27 |        55.3  |         0.34 |        65.6  |         0.41 |
| Precision      | FP32 (Ref)             |       14.9 |         32.1 |  31.2 |        76.1  |         0.28 |        55.1  |         0.35 |        65.4  |         0.42 |
| Precision      | FP16 AMP               |       14.9 |         22.3 |  44.8 |        76.05 |         0.29 |        55.05 |         0.36 |        65.35 |         0.43 |

### Table 3: nuScenes-mini Surround-View Benchmark

| model                    | camera_backbone   |   voxel_size |   seeds |   mAP |   mAP_std |   NDS |   NDS_std |   latency_ms_t4 |   fps_t4 |
|:-------------------------|:------------------|-------------:|--------:|------:|----------:|------:|----------:|----------------:|---------:|
| PointPillars_LiDAR_Only  | nan               |          0.2 |   0_1_2 | 29.4  |      0.45 | 38.6  |      0.52 |            18.2 |     54.9 |
| BEVFusionLite_Heavy      | ResNet-50         |          0.2 |   0_1_2 | 34.2  |      0.4  | 44.8  |      0.48 |            86.4 |     11.6 |
| BEVFusionLite_Light_FP32 | ResNet-18         |          0.4 |   0_1_2 | 31.8  |      0.38 | 42.1  |      0.44 |            38.5 |     26   |
| BEVFusionLite_Light_FP16 | ResNet-18         |          0.4 |   0_1_2 | 31.75 |      0.39 | 42.05 |      0.45 |            26.8 |     37.3 |

