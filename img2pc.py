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

# 1. Load image
image_path = "1.png"  # Path to your input image
image = Image.open(image_path).convert("RGB")

# convert image to a NumPy array so we can attach color to each 3D point
rgb_image = np.array(image) # (H, W, 3) uint8
h, w = rgb_image.shape[:2]

# 2. Run monocular depth estimation
device = 0 if torch.cuda.is_available() else -1

# hugging face "depth-estimation" pipeline returns a predicted depth tensor/map
# Depth Anything V2 typically produces *relative depth*: good geometry but unknown scale
depth_pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-base-hf",
    device=device,
)

out = depth_pipe(image)

# the pipeline returns a "preicted_depth"
# depending on versions, this might be a torch.Tensor already
pred = out["predicted_depth"]
depth_t = pred.float() if isinstance(pred, torch.Tensor) else torch.tensor(np.asarray(pred), dtype=torch.float32)

# 3. Make depth match the image resolution (if needed)
# many models output depth at a different resolution than the input image
# we resize the depth map to (h, w) so each pixel has a depth value
depth_t = depth_t.unsqueeze(0).unsqueeze(0)  # (1, 1, H_d, W_d)
depth_t = F.interpolate(depth_t, size=(h, w), mode="bilinear", align_corners=False)
depth_t = depth_t.squeeze(0).squeeze(0)  # (H, W)

# convert depth tensor to NumPy array for the geometry step
D = depth_t.cpu().numpy().astype(np.float32)  # (H, W) float32

# depth should be positive, clip small/negative values
D = np.clip(D, a_min=1e-3, a_max=None)

# 4. Camera intrinsics (fx, fy, cx, cy)
# to back-project pixels into 3D we need camera intrinsics
# best practice: calibrate your camera to get accurate fx, fy, cx, cy

# if you don't have calibration data, you can approximate fx, fy using assumed FOV
# this makes a "shaped" point cloud that looks correct but is not metrically accurate
fov_deg = 60.0  # assumed horizontal field of view in degrees
fx = fy = 0.5 * w / np.tan(0.5 * np.deg2rad(fov_deg / 2.0))

# principal point at image ~center
cx = w / 2.0
cy = h / 2.0

# 5. Back-project pixels to 3D points
# create a grid of (u, v) pixel coordinates
# u: x-coordinates (0...w-1), v: y-coordinates (0...h-1)
u, v = np.meshgrid(np.arange(w), np.arange(h))  # both (H, W)

# interpret depth D as Z for each pixel
Z = D  # (H, W)

# apply pinhole camera back-projection
X = (u - cx) * Z / fx  # (H, W)
Y = (v - cy) * Z / fy  # (H, W)

# stack X, Y, Z into (N, 3) point cloud where N = H * W
xyz = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)  # (N, 3)

# grab a matching color for each 3D point
colors = rgb_image.reshape(-1, 3)  # (N, 3)

# 6. Optional cleanup: remove points with very large depth (e.g., sky)
# with relative depth maps, large depth values are often unreliable
# removing tails can improve visualization
z = xyz[:, 2]   
z_lo, z_hi = np.percentile(z, 5), np.percentile(z, 95)
keep = (z >= z_lo) & (z <= z_hi)

xyz = xyz[keep]
colors = colors[keep]

# 7. Write point cloud to PLY file
ply_path = "output_point_cloud.ply"
write_ply(ply_path, xyz, colors)
print(f"Wrote point cloud with {xyz.shape[0]} points to {ply_path}")

"""
How to view:
- Copy out.ply to another machine and open in MeshLab / CloudCompare / Blender.
- On Pi you can still use MeshLab sometimes, but it can be slow.

If you want this to be closer to "real-world scale":
- Use a metric depth model (if it fits your use case), OR
- Calibrate your camera to get accurate intrinsics, OR
- Provide a known scale reference in the scene and rescale Z accordingly.
"""