from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from transformers import pipeline
from pathlib import Path

class PointCloud:
    def __init__(self, image_path="", model="depth-anything/Depth-Anything-V2-base-hf"):
        self.image_path = image_path
        self.model = model
        self.rgb_image = self.load_image()
        self.cam_intrinsics = self.estimate_depth() # fx, fy, cx, cy, D
        self.colors, self.backprojected_points = self.backproject_points(cleanup=True)
    

    def load_image(self):
        """
        Load the input image from disk.
        
        :param self: PointCloud object

        :return: RGB image as a (H, W, 3) NumPy array
        """
        image = Image.open(self.image_path).convert("RGB")

        # resize to keep inputs reasonably small (preserve aspect ratio).
        max_side = max(image.size)
        if max_side > 1024:
            scale = 1024 / max_side
            new_w = max(1, int(round(image.size[0] * scale)))
            new_h = max(1, int(round(image.size[1] * scale)))
            image = image.resize((new_w, new_h), resample=Image.LANCZOS)
        
        # convert image to a NumPy array so we can attach color to each 3D point
        rgb_image = np.array(image) # (H, W, 3) uint8
        return rgb_image


    def estimate_depth(self, focal_length=35.0, sensor_in=(1.0/1.56)):
        """
        Use a depth estimation model to predict the depth map from the image.
        
        :param self: PointCloud object
        :param focal_length: camera focal length in mm
        :param sensor_in: diagonal camera sensor size in inches

        :return: fx, fy, cx, cy, D where D is the depth map as a (H, W) NumPy array and fx, fy, cx, cy are camera intrinsics
        """
        h, w = self.rgb_image.shape[:2]

        device = 0 if torch.cuda.is_available() else -1
        
        # hugging face "depth-estimation" pipeline returns a predicted depth tensor/map
        # Depth Anything V2 typically produces *relative depth*: good geometry but unknown scale
        depth_pipe = pipeline(
            task="depth-estimation",
            model=self.model,
            device=device,
            image_processor_kwargs={"use_fast": True}
        )

        # depth map
        out = depth_pipe(self.image_path)        
        pred = out["predicted_depth"]

        # store depth as float tensor
        self.depth_t = pred.float() if isinstance(pred, torch.Tensor) else torch.tensor(pred).float()

        # make depth match the image resolution
        depth_t = self.depth_t.unsqueeze(0).unsqueeze(0)  # (1, 1, H_d, W_d)
        depth_t = F.interpolate(depth_t, size=(h, w), mode="bilinear", align_corners=False)
        depth_t = depth_t.squeeze(0).squeeze(0)  # (H, W)

        # convert depth tensor to NumPy array for the geometry step
        D = depth_t.cpu().numpy().astype(np.float32)  # (H, W) float32
        # D = np.clip(D, a_min=0, a_max=None)  # ensure non-negative depths

        # camera intrinsics
        w_px, h_px = w, h
        aspect = w_px / h_px

        # convert optical format to actual sensor diagonal (mm)
        sensor_diag = sensor_in * 25.4  # inches to mm
        sensor_h = np.sqrt((sensor_diag ** 2) / (aspect ** 2 + 1))
        sensor_w = aspect * sensor_h

        fx = (focal_length * w_px) / sensor_w
        fy = (focal_length * h_px) / sensor_h
        cx = w_px / 2.0
        cy = h_px / 2.0

        return fx, fy, cx, cy, D
    

    def backproject_points(self, cleanup):
        """
        Create a grid of 3D points from the depth map and camera intrinsics.

        :self: PointCloud object
        :cleanup: whether to filter out points with very large depth values

        :return: colors (N, 3) and xyz (N, 3) point cloud
        """
        fx, fy, cx, cy, D = self.cam_intrinsics
        h, w = D.shape
        
        # create a grid of (u, v) pixel coordinates
        u, v = np.meshgrid(np.arange(w), np.arange(h))  # (H, W)
        
        # apply pinhole camera back-projection
        # image v grows downward, so negate Y to make +Y up in camera space
        # Flip depth so the cloud faces the default viewer camera direction.
        Z = -D
        X = (u - cx) * Z / fx
        Y = -(v - cy) * Z / fy

        # stack X,Y,Z into (N, 3) point cloud where N = H * W
        xyz = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)  # (H, W, 3)

        # grab the matching color for each 3D point
        colors = self.rgb_image.reshape(-1, 3)  # (H, W, 3) -> (N, 3)

        # optional cleanup, filter out points with very large depth values
        # with relative depth maps this is a heuristic to remove outliers
        if cleanup:
            z = xyz[:, 2]   
            z_hi = np.percentile(z, 95)
            keep = (z <= z_hi)

            xyz = xyz[keep]
            colors = colors[keep]

        return colors, xyz
    

    def write_ply(self, destination):
        """
        Write the point cloud to a PLY file.

        :param self: PointCloud object
        :param destination: output PLY file path location
        """
        ply_path = destination + Path(self.image_path).stem + "".join(self.model.split("/")) + ".ply"

        n = self.backprojected_points.shape[0]

        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header"
        ]

        with open(ply_path, "w") as f:
            f.write("\n".join(header) + "\n")
            for i in range(n):
                x, y, z = self.backprojected_points[i]
                r, g, b = self.colors[i]
                f.write(f"{x} {y} {z} {r} {g} {b}\n")
        
        print(f"Wrote point cloud with {n} points to {ply_path}")
