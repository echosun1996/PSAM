import numpy as np
import pandas as pd
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import matplotlib.pyplot as plt
import json
import os
from PIL import Image, ImageDraw
import cv2
from shapely import point_on_surface
import torch
import logging
from time import time
import csv
import argparse
import sys
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(current_dir), "py_scripts/final/"))

from calculate_metrics import Dice


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, help="checkpoint")
parser.add_argument("--input-path", type=str, help="input path")
parser.add_argument("--output-path", type=str, help="output path")
parser.add_argument("--gt-path", type=str, help="ground truth path")
parser.add_argument("--point-csv-path", type=str, help="point csv path", required=True)

args = parser.parse_args()


sam = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
ground_truth_path = args.gt_path
point_csv_path = args.point_csv_path


# 检测 cuda 是否可用
def gpu_available():
    flag = torch.cuda.is_available()
    if flag:
        logging.info("CUDA可使用")
        return True
    else:
        logging.warning("CUDA不可用")
        return False


# 检测GPU可用性
if gpu_available():
    logging.info("载入SAM模型")
    sam.to(device="cuda")


# 修改现有函数以返回掩膜
def generate_sam_entire_masks(sam_para, image_path):
    mask_generator = SamAutomaticMaskGenerator(sam_para)
    # 加载原始图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 生成掩膜
    masks = mask_generator.generate(image)
    return masks


def find_and_save_mask(anns, ground_truth_tensor, filename):
    dice = Dice()
    h, w = ground_truth_tensor.shape[-2:]
    black_mask_image = np.zeros((h, w, 3), dtype=np.uint8)
    color = np.array([255, 255, 255])

    max_mask = None
    max_mask_score = -1

    mask = anns
    predicted_array = mask.astype(int)
    predicted_unique_values = np.unique(predicted_array)
    if set(predicted_unique_values).issubset({0}):
        print("Predicted mask is all zeros")
    elif not set(predicted_unique_values).issubset({0, 1}):
        print("Predicted mask is not binary:", set(predicted_unique_values))
        exit(-1)

    predicted_tensor = torch.tensor(predicted_array, dtype=torch.float32)

    dice_score = dice(predicted_tensor, ground_truth_tensor)
    if dice_score > max_mask_score:
        max_mask_score = dice_score
        max_mask = mask

    if max_mask is None:
        mask_img = Image.fromarray(black_mask_image)
        mask_img.save(filename)
    else:
        black_mask_image[max_mask == 1] = color
        mask_img = Image.fromarray(black_mask_image)
        mask_img.save(filename)


def save_timecost(img_name, timecost, csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write("img_name,timecost\n")
    df = pd.read_csv(csv_path)
    if img_name in df["img_name"].values:
        return
    with open(csv_path, "a") as f:
        f.write(f"{img_name},{timecost}\n")


def read_front_prompt(image_file_name, pa_point_csv_path):

    coordinates_list = []
    labels = []
    shape = []
    # 打开CSV文件并搜索匹配的行
    with open(pa_point_csv_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["img_filename"] == image_file_name:
                # 将每个坐标点添加到列表中
                coordinates = []
                for i in range(100):
                    if f"x_{i}" not in row or f"y_{i}" is None:
                        break
                    y = float(row[f"x_{i}"])
                    x = float(row[f"y_{i}"])
                    coordinates.append([x, y])
                    labels.append(1)
                coordinates_list.append(coordinates)

    if len(coordinates_list) != 1:
        logging.error("Find more than 1 coordinates.")
        exit(-1)

    coord_ponts = coordinates_list[0]
    coord_lables = labels
    # print(coord_ponts)
    # print(coord_lables)

    return np.array(coord_ponts), np.array(coord_lables)


def generate_sam_from_point(sam_para, image_path, input_point, input_label):
    predictor = SamPredictor(sam_para)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=input_point, point_labels=input_label, multimask_output=True
    )
    max_score_index = np.argmax(scores)
    max_score_mask = masks[max_score_index]
    return max_score_mask


input_folder = args.input_path
processing_name = input_folder.split("/")[-1]
timecost_path = os.path.join(args.output_path, "SAMSpecific_timecost.csv")
output_path = os.path.join(args.output_path, "SAMSpecific")
os.makedirs(output_path, exist_ok=True)
if os.path.exists(timecost_path):
    os.remove(timecost_path)

for image_file in tqdm(os.listdir(input_folder), desc="Processing " + processing_name):
    image_path = os.path.join(input_folder, image_file)
    gt_path = os.path.join(
        ground_truth_path, image_file.split(".")[0] + "_segmentation.png"
    )
    input_point, input_label = read_front_prompt(image_file, point_csv_path)
    start_time = time()
    predicted_anns = generate_sam_from_point(sam, image_path, input_point, input_label)
    end_time = time()
    time_costs = end_time - start_time

    ground_truth_image = Image.open(gt_path).convert("L")
    ground_truth_array = np.array(ground_truth_image)
    ground_truth_unique_values = np.unique(ground_truth_array)
    if not set(ground_truth_unique_values).issubset({0, 1}) or set(
        ground_truth_unique_values
    ).issubset({0}):
        ground_truth_array = (ground_truth_array > 127).astype(np.uint8)
        ground_truth_unique_values = np.unique(ground_truth_array)
    else:
        print(
            "Ground truth:" + gt_path + " is binary:",
            set(ground_truth_unique_values),
        )
        print("Ground truth not visable, run 'binary2visable.py' first!")
        exit(-1)
    ground_truth_tensor = torch.tensor(ground_truth_array, dtype=torch.float32)

    find_and_save_mask(
        predicted_anns, ground_truth_tensor, os.path.join(output_path, image_file)
    )
    save_timecost(image_file, time_costs, timecost_path)
