"""
BEVFusionLite model components.

Modules:
    - camera_backbone: ResNet-18 + FPN for image feature extraction
    - lss_transform: Lift-Splat-Shoot view transformer (camera → BEV)
    - depth_net: Depth distribution prediction network
    - bev_fusion: BEV feature fusion (camera + LiDAR)
    - bevfusionlite_detector: Main detector class (OpenPCDet plugin)
"""
