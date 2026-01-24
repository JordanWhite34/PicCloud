# PicCloud

Generate simple point clouds from a single image using Hugging Face depth-estimation models.

## What it does
- Loads an RGB image
- Estimates a depth map with a Hugging Face `depth-estimation` pipeline
- Backprojects pixels into 3D points
- Writes an ASCII `.ply` point cloud with per-point color

## Requirements
- Python 3.9+ recommended
- See `requirements.txt` for Python dependencies
- Optional: CUDA for faster inference

## Project layout
- `point_cloud.py`: core `PointCloud` class
- `usage.py`: example usage (loops through a few models)
- `inputs/`: place input images here
- `outputs/`: generated `.ply` files

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Put an image in `inputs/` (e.g. `test.jpg`), then run:
```bash
python usage.py
```

Or use the class directly:
```python
from point_cloud import PointCloud

pc = PointCloud(image_path="inputs/test.jpg", model="depth-anything/Depth-Anything-V2-base-hf")
pc.write_ply(destination="outputs/")
```

## Notes
- Many depth models output *relative* depth; scale is not metric.
- The output filename combines the image stem and model name.
- Large images are resized so the longest side is 1024px.

## License
MIT.
