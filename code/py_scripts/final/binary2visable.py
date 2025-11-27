from importlib.metadata import files
import cv2
import numpy as np
from PIL import Image
import os
import argparse
from tqdm import tqdm
from rich import print

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default=None, help="ground truth path")

args = parser.parse_args()

process_path = args.path
print(f"Processing path to visible mask: {process_path}")
for file in tqdm(os.listdir(process_path)):
    if file.split(".")[-1] != "png":
        continue
    predicted_path = os.path.join(process_path, file)
    predicted_image = Image.open(predicted_path).convert("L")
    predicted_array = np.array(predicted_image)
    predicted_unique_values = np.unique(predicted_array)
    if set(predicted_unique_values).issubset({0, 1}):
        predicted_array = predicted_array * 255
    cv2.imwrite(predicted_path, predicted_array)
