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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# from code.compare_models.reps.OnePrompt import dataset
from scribble.scribbles import (
    LineScribble,
    CenterlineScribble,
    ContourScribble,
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
lineScribble_path = os.path.join(
    output_folder, dataset_name + "_positive_lineScribble_" + str(labeller)
)
if os.path.exists(lineScribble_path):
    shutil.rmtree(lineScribble_path)
os.makedirs(lineScribble_path, exist_ok=True)

n_lineScribble_path = os.path.join(
    output_folder, dataset_name + "_negative_lineScribble_" + str(labeller)
)
if os.path.exists(n_lineScribble_path):
    shutil.rmtree(n_lineScribble_path)
os.makedirs(n_lineScribble_path, exist_ok=True)

lineScribble = LineScribble()

# centerlineScribble
centerlineScribble_path = os.path.join(
    output_folder, dataset_name + "_positive_centerlineScribble_" + str(labeller)
)
if os.path.exists(centerlineScribble_path):
    shutil.rmtree(centerlineScribble_path)
os.makedirs(centerlineScribble_path, exist_ok=True)

n_centerlineScribble_path = os.path.join(
    output_folder, dataset_name + "_negative_centerlineScribble_" + str(labeller)
)
if os.path.exists(n_centerlineScribble_path):
    shutil.rmtree(n_centerlineScribble_path)
os.makedirs(n_centerlineScribble_path, exist_ok=True)
centerlineScribble = CenterlineScribble()


# contourScribble
contourScribble_path = os.path.join(
    output_folder, dataset_name + "_positive_contourScribble_" + str(labeller)
)
if os.path.exists(contourScribble_path):
    shutil.rmtree(contourScribble_path)
os.makedirs(contourScribble_path, exist_ok=True)

n_contourScribble_path = os.path.join(
    output_folder, dataset_name + "_negative_contourScribble_" + str(labeller)
)
if os.path.exists(n_contourScribble_path):
    shutil.rmtree(n_contourScribble_path)
os.makedirs(n_contourScribble_path, exist_ok=True)
contourScribble = ContourScribble()

transform = transforms.Compose([transforms.ToTensor()])

for filename in tqdm(os.listdir(input_folder)):
    if filename.lower().endswith(("_segmentation.png")):
        output_filename = filename.replace("_segmentation.png", ".png")
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale

        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Generate scribbles
        lineScribbles_tensor = lineScribble(image_tensor, n_scribbles=1)
        centerlineScribbles_tensor = centerlineScribble(image_tensor, n_scribbles=1)
        contourScribbles_tensor = contourScribble(image_tensor, n_scribbles=1)

        # 生成负面涂鸦的掩码
        negative_mask = (image_tensor == 0).float()

        n_lineScribbles_tensor = lineScribble(negative_mask, n_scribbles=1)
        n_centerlineScribbles_tensor = centerlineScribble(negative_mask, n_scribbles=1)
        n_contourScribbles_tensor = contourScribble(negative_mask, n_scribbles=1)

        # Convert tensor to numpy array
        lineScribbles_np = lineScribbles_tensor.squeeze().cpu().numpy() * 255
        lineScribbles_np = lineScribbles_np.astype(np.uint8)
        centerlineScribbles_np = (
            centerlineScribbles_tensor.squeeze().cpu().numpy() * 255
        )
        centerlineScribbles_np = centerlineScribbles_np.astype(np.uint8)
        contourScribbles_np = contourScribbles_tensor.squeeze().cpu().numpy() * 255
        contourScribbles_np = contourScribbles_np.astype(np.uint8)

        n_lineScribbles_np = n_lineScribbles_tensor.squeeze().cpu().numpy() * 255
        n_lineScribbles_np = n_lineScribbles_np.astype(np.uint8)
        n_centerlineScribbles_np = (
            n_centerlineScribbles_tensor.squeeze().cpu().numpy() * 255
        )
        n_centerlineScribbles_np = n_centerlineScribbles_np.astype(np.uint8)
        n_contourScribbles_np = n_contourScribbles_tensor.squeeze().cpu().numpy() * 255
        n_contourScribbles_np = n_contourScribbles_np.astype(np.uint8)

        # Save scribbles image
        output_path = os.path.join(lineScribble_path, output_filename)
        cv2.imwrite(output_path, lineScribbles_np)
        output_path = os.path.join(centerlineScribble_path, output_filename)
        cv2.imwrite(output_path, centerlineScribbles_np)
        output_path = os.path.join(contourScribble_path, output_filename)
        cv2.imwrite(output_path, contourScribbles_np)

        output_path = os.path.join(n_lineScribble_path, output_filename)
        cv2.imwrite(output_path, n_lineScribbles_np)
        output_path = os.path.join(n_centerlineScribble_path, output_filename)
        cv2.imwrite(output_path, n_centerlineScribbles_np)
        output_path = os.path.join(n_contourScribble_path, output_filename)
        cv2.imwrite(output_path, n_contourScribbles_np)
