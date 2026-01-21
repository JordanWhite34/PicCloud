from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from transformers import pipeline

class PointCloud:
    def __init__(self, image_path="", model="depth-anything/Depth-Anything-V2-base-hf"):
        self.image_path = image_path
        self.model = model
        self.load_image()
        self.estimate_depth()
    
    def load_image(self):
        self.image = Image.open(self.image_path).convert("RGB")

        # resize to keep inputs reasonably small (preserve aspect ratio).
        max_side = max(self.image.size)
        if max_side > 1024:
            scale = 1024 / max_side
            new_w = max(1, int(round(self.image.size[0] * scale)))
            new_h = max(1, int(round(self.image.size[1] * scale)))
            self.image = self.image.resize((new_w, new_h), resample=Image.LANCZOS)

    
    def estimate_depth(self):
        # convert image to a NumPy array so we can attach color to each 3D point
        rgb_image = np.array(self.image) # (H, W, 3) uint8
        h, w = rgb_image.shape[:2]
        # TODO: make sure this can handle non-default models
        device = 0 if torch.cuda.is_available() else -1
        
        # hugging face "depth-estimation" pipeline returns a predicted depth tensor/map
        # Depth Anything V2 typically produces *relative depth*: good geometry but unknown scale
        depth_pipe = pipeline(
            task="depth-estimation",
            model=self.model,
            device=device,
        )

        # depth map
        out = depth_pipe(self.image)        
        pred = out["predicted_depth"]

        # store depth as float tensor
        self.depth_t = pred.float() if isinstance(pred, torch.Tensor) else torch.tensor(pred).float()

        # make depth match the image resolution
        depth_t = self.depth_t.unsqueeze(0).unsqueeze(0)  # (1, 1, H_d, W_d)
        depth_t = F.interpolate(depth_t, size=(h, w), mode="bilinear", align_corners=False)