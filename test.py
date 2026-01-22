from point_cloud import PointCloud

image_path = "test.jpg"

pc = PointCloud(image_path="inputs/" + image_path)
pc.write_ply(destination="outputs/")