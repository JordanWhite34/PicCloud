from point_cloud import PointCloud

image_path = "2.png"

# example models to test
models_to_test = [
    "Intel/dpt-swinv2-tiny-256",
    "Intel/dpt-hybrid-midas",
    "LiheYoung/depth-anything-small-hf",
    "depth-anything/Depth-Anything-V2-Small-hf",
]

for model_name in models_to_test:
    print(f"Generating point cloud for model: {model_name}")
    pc = PointCloud(image_path="inputs/" + image_path, model=model_name)
    pc.write_ply(destination="outputs/")
