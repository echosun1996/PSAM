import os
import sys
import argparse
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default=None, help="input path")
parser.add_argument("--output", type=str, default=None, help="output path")

args = parser.parse_args()
input_path = args.input
output_path = args.output
print("Processing binary mask to binary image:", input_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)

for file in os.listdir(input_path):
    img = Image.open(os.path.join(input_path, file))
    img_array = np.array(img)
    binary_mask = (img_array > 0).astype(np.uint8)
    binary_mask_img = Image.fromarray(binary_mask)
    binary_mask_img.save(os.path.join(output_path, file))
