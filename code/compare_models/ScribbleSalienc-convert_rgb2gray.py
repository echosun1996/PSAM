from PIL import Image
import numpy as np
import os

import argparse

from sympy import im
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--img-input", type=str, help="img input path")
parser.add_argument("--gray-output", type=str, help="output gray path")

args = parser.parse_args()


def gamma_expansion(image):
    """Apply gamma expansion to an image array"""
    gamma = 2.2  # Example gamma value; adjust as needed
    return np.power(image / 255.0, gamma)


# Define directories
# img_dir = "/home/jing-zhang/jing_file/RGB_sal_dataset/train/DUTS/img/"
# save_img_dir = "./gray/"
img_dir = args.img_input
save_img_dir = args.gray_output


# Ensure output directory exists
os.makedirs(save_img_dir, exist_ok=True)

# Process each image
for filename in tqdm(os.listdir(img_dir), desc=img_dir.split("/")[-1]):
    if filename.endswith(".jpg"):
        # print(f"Processing: {filename}")
        # Read image
        img_cur = Image.open(os.path.join(img_dir, filename))
        img_cur = img_cur.convert("RGB")  # Ensure image is in RGB format

        # Convert to numpy array
        img_array = np.array(img_cur)

        if img_array.ndim == 3 and img_array.shape[2] == 3:
            img_r = img_array[:, :, 0]
            img_g = img_array[:, :, 1]
            img_b = img_array[:, :, 2]
        elif img_array.ndim == 2:
            img_r = img_g = img_b = img_array
        else:
            raise ValueError("Unsupported image format")

        # Apply gamma expansion
        img_r_linear = gamma_expansion(img_r)
        img_g_linear = gamma_expansion(img_g)
        img_b_linear = gamma_expansion(img_b)

        # Convert to grayscale using the specified weights
        gray_cur = 0.2126 * img_r_linear + 0.7152 * img_g_linear + 0.0722 * img_b_linear

        # Convert to 8-bit image
        gray_cur_uint8 = np.uint8(gray_cur * 255)

        # Save grayscale image
        gray_img = Image.fromarray(gray_cur_uint8)
        gray_img.save(
            os.path.join(save_img_dir, f"{os.path.splitext(filename)[0]}.png")
        )
