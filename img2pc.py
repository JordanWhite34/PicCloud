"""
Single-image -> depth -> point cloud (PLY) on raspberry pi os (no Open3D)

What this script does:
1) Loads an RGB image
2) Uses a pretrained monocular depth estimation model to predict a depth map
   - For single image models like Depth Anything, the depth is usually relative (scale is arbitrary)
3) Converts each pixel (u, v) with depth z into a 3D point (X, Y, Z) using a pinhole camera model
4) Writes the resulting colored point cloud to an ASCII .ply file you can open in MeshLab or CloudCompare

Key math (pinhole back-projection):
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = depth(u, v)
Where:
- (u, v) are pixel coordinates
- (cx, cy) is the principal point (image center)
- (fx, fy) are focal lengths in pixels
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import pipeline

def write_ply(path: str, xyz: np.ndarray, rgb: np.ndarray | None = None) -> None:
    """
    Write a point cloud in ASCII PLY format.

    xyz: (N, 3) float32/float64 array of 3D points (X, Y, Z)
    rgb: (N, 3) uint8 array of colors (R, G, B) in [0, 255], optional

    PLY is a simple text/binary format for 3D data, supported by many 3D viewers.
    """
    n = xyz.shape[0]
    has_rgb = rgb is not None

    # PLY header describes how many vertices and which properties each vertex has
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]

    if has_rgb:
        header += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header.append("end_header")

    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        
        # each following line is one vertex
        if has_rgb:
            for (x, y, z), (r, g, b) in zip(xyz, rgb):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
        else:
            for (x, y, z) in xyz:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")