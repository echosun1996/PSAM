import os
import random
import shutil
import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from rich import print
import sys
import os

from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# from code.compare_models.reps.OnePrompt import dataset
from scribble.scribbles import (
    LineScribble,
    CenterlineScribble,
    ContourScribble,
    XScribble,
)  # Replace 'your_module' with the actual module name where LineScribble is defined

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-name", type=str, default=None, help="dataset name")

parser.add_argument(
    "--input-path", type=str, default=None, help="ground truth input path"
)
parser.add_argument("--output-path", type=str, default=None, help="scibble output path")
parser.add_argument("--labeller", type=int, default=42, help="labeller")

args = parser.parse_args()

# Set paths
input_folder = args.input_path
output_folder = args.output_path
dataset_name = args.dataset_name
labeller = args.labeller
random.seed(labeller)
np.random.seed(labeller)

print(f"Start genenate scibble: {dataset_name}")

# lineScribble
xScribble_path = os.path.join(
    output_folder, dataset_name + "_positive_xScribble_" + str(labeller)
)
if os.path.exists(xScribble_path):
    shutil.rmtree(xScribble_path)
os.makedirs(xScribble_path, exist_ok=True)


xscribble = XScribble()
transform = transforms.Compose([transforms.ToTensor()])

for filename in tqdm(os.listdir(input_folder)):
    if filename.lower().endswith(("_segmentation.png")):

        output_filename = filename.replace("_segmentation.png", ".png")
        image_path = os.path.join(input_folder, filename)
        # print(f"Input image: {image_path}")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale

        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        xScribbles_tensor = xscribble(image_tensor)
        xScribbles_np = xScribbles_tensor.squeeze().cpu().numpy() * 255
        xScribbles_np = xScribbles_np.astype(np.uint8)

        # Save scribbles image
        output_path = os.path.join(xScribble_path, output_filename)
        cv2.imwrite(output_path, xScribbles_np)
        # print(f"Saved scribble image to {output_path}")

        # exit(-1)
