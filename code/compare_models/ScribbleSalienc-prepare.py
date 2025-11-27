from PIL import Image
import os
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--positive-scribble-input", type=str, help="positive scribble input path"
)
parser.add_argument(
    "--neagtive-scribble-input", type=str, help="neagtive scribble input path"
)
parser.add_argument("--mask-output", type=str, help="output mask path")
parser.add_argument("--gt-output", type=str, help="output gt path")

args = parser.parse_args()


# Define directories
# scribble_dir = "/home/jing-zhang/jing_file/RGB_sal_dataset/scribbled/scribble/"
# mask_dir = "/home/jing-zhang/jing_file/RGB_sal_dataset/scribbled/mask/"
# new_gt_dir = "/home/jing-zhang/jing_file/RGB_sal_dataset/scribbled/gt/"

positive_scribble_dir = args.positive_scribble_input
neagtive_scribble_dir = args.neagtive_scribble_input
mask_dir = args.mask_output
new_gt_dir = args.gt_output

# Ensure output directories exist
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(new_gt_dir, exist_ok=True)

# Process each image
for filename in tqdm(
    os.listdir(positive_scribble_dir),
    desc=positive_scribble_dir.split("/")[-1],
):
    if filename.endswith(".png"):
        # print(f"Processing: {filename}")
        # Read image
        positive_img_cur = Image.open(os.path.join(positive_scribble_dir, filename))
        positive_img_cur = positive_img_cur.convert(
            "L"
        )  # Convert to grayscale if not already

        negative_img_cur = Image.open(os.path.join(neagtive_scribble_dir, filename))
        negative_img_cur = negative_img_cur.convert(
            "L"
        )  # Convert to grayscale if not already

        # Get image size
        h, w = positive_img_cur.size

        # Create masks
        mask_cur = Image.new("L", (w, h), 0)
        new_scri = Image.new("L", (w, h), 0)

        positive_pixels = positive_img_cur.load()
        negative_pixels = negative_img_cur.load()
        mask_pixels = mask_cur.load()
        new_scri_pixels = new_scri.load()

        for y in range(h):
            for x in range(w):
                if positive_pixels[x, y] != 0:
                    mask_pixels[x, y] = 255
                    new_scri_pixels[x, y] = 255
                if negative_pixels[x, y] != 0:
                    mask_pixels[x, y] = 255

        # Save images
        mask_cur.save(os.path.join(mask_dir, filename))
        new_scri.save(os.path.join(new_gt_dir, filename))
