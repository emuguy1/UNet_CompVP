from PIL import Image
from pathlib import Path

# Define the directories and parameters
images_dir = Path('/home/cip/2022/ce90tate/UNet_CompVP/data/training_set/cloth/')
distances_dir = Path('/home/cip/2022/ce90tate/UNet_CompVP/data/training_set/distances/')
expected_dims = (512, 512)  # Expected dimensions (width, height)
expected_channels = 1  # Expected number of channels (1 for grayscale)

# Iterate over the range of images
for i in range(1, 101):  # Adjust the range as per your requirements
    for j in range(1, 51):
        image_name = f'{i}_{j}.png'
        image_path = images_dir / image_name
        distance_path = distances_dir / image_name

        # Check if both files exist
        if not image_path.is_file() or not distance_path.is_file():
            print(f"Missing file: {image_name}")
            continue

        # Open and check dimensions and channels
        with Image.open(image_path) as img, Image.open(distance_path) as dist:
            if img.size != expected_dims or dist.size != expected_dims:
                print(
                    f"Dimension mismatch in {image_name}: Image dimensions {img.size}, Distance dimensions {dist.size}")

            if len(img.getbands()) != expected_channels or len(dist.getbands()) != expected_channels:
                print(
                    f"Channel mismatch in {image_name}: Image channels {len(img.getbands())}, Distance channels {len(dist.getbands())}")
