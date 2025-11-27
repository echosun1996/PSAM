import csv
from rich import print
import os
from regex import P
from sklearn.metrics import jaccard_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse


class Sensitivity(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Sensitivity, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives and False Negatives
        true_positive = (inputs * targets).sum()
        false_negative = ((1 - inputs) * targets).sum()

        sensitivity = (true_positive + smooth) / (
            true_positive + false_negative + smooth
        )

        return sensitivity


class Specificity(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Specificity, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Negatives and False Positives
        true_negative = ((1 - inputs) * (1 - targets)).sum()
        false_positive = (inputs * (1 - targets)).sum()

        specificity = (true_negative + smooth) / (
            true_negative + false_positive + smooth
        )
        return specificity


class Accuracy(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Accuracy, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, True Negatives, False Positives, and False Negatives
        true_positive = (inputs * targets).sum()
        true_negative = ((1 - inputs) * (1 - targets)).sum()
        false_positive = (inputs * (1 - targets)).sum()
        false_negative = ((1 - inputs) * targets).sum()

        accuracy = (true_positive + true_negative + smooth) / (
            true_positive + true_negative + false_positive + false_negative + smooth
        )

        return accuracy


# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook#BCE-Dice-Loss
class Jaccard(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Jaccard, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return IoU


class Dice(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return dice


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run nnUNet to predict masks")

    parser.add_argument("--gt-path", type=str, help="ground truth path", default="")
    parser.add_argument(
        "--predict-output", type=str, help="predict output path", default=""
    )
    parser.add_argument(
        "--csv-output-path", type=str, help="csv output path", default=""
    )
    parser.add_argument("--image-size", nargs=2, default=[1024, 1024], type=int)

    args = parser.parse_args()

    input_size = args.image_size

    true_mask_path = os.path.join(args.gt_path)
    predict_output = os.path.join(args.predict_output)
    csv_output_path = os.path.join(args.csv_output_path)

    seg_files = os.listdir(predict_output)
    seg_files.sort()

    def calculate_metrics_from_path(predicted_path, groud_truth_path):
        predicted_image = Image.open(predicted_path).convert("L")
        ground_truth_image = Image.open(groud_truth_path).convert("L")

        # Convert images to NumPy arrays
        predicted_array = np.array(predicted_image)
        ground_truth_array = np.array(ground_truth_image)
        predicted_unique_values = np.unique(predicted_array)
        ground_truth_unique_values = np.unique(ground_truth_array)

        if not set(predicted_unique_values).issubset({0, 1}) or set(
            predicted_unique_values
        ).issubset({0}):
            predicted_array = (predicted_array > 127).astype(np.uint8)
            predicted_unique_values = np.unique(predicted_array)
        else:
            print(
                "Predicted mask:"
                + predicted_path
                + " is binary:"
                + set(predicted_unique_values),
            )
            print("Predicted mask not visable, run 'binary2visable.py' first!")
            exit(-1)
        if not set(ground_truth_unique_values).issubset({0, 1}) or set(
            ground_truth_unique_values
        ).issubset({0}):
            ground_truth_array = (ground_truth_array > 127).astype(np.uint8)
            ground_truth_unique_values = np.unique(ground_truth_array)
        else:
            print(
                "Ground truth:" + groud_truth_path + " is binary:",
                set(ground_truth_unique_values),
            )
            print("Ground truth not visable, run 'binary2visable.py' first!")
            exit(-1)

        assert (input_size == list(predicted_array.shape)) and (
            input_size == list(ground_truth_array.shape)
        ), f"Image size is not match ({input_size[0]} x {input_size[1]})"

        assert (
            predicted_array.shape == ground_truth_array.shape
        ), "predicted_array & ground_truth_array shape do not match"

        assert set(ground_truth_unique_values).issubset(
            {0, 1}
        ), "Ground truth image is not binary"
        assert set(predicted_unique_values).issubset(
            {0, 1}
        ), "Predicted image is not binary"

        predicted_tensor = torch.tensor(predicted_array, dtype=torch.float32)
        ground_truth_tensor = torch.tensor(ground_truth_array, dtype=torch.float32)

        dice = Dice()
        dice_score = dice(predicted_tensor, ground_truth_tensor)

        jaccard = Jaccard()
        jaccard_score = jaccard(predicted_tensor, ground_truth_tensor)

        sensitivity = Sensitivity()
        sensitivity_score = sensitivity(predicted_tensor, ground_truth_tensor)

        specificity = Specificity()
        specificity_score = specificity(predicted_tensor, ground_truth_tensor)

        accuracy = Accuracy()
        accuracy_score = accuracy(predicted_tensor, ground_truth_tensor)

        return {
            "jaccard_score": jaccard_score.item(),
            "dice_score": dice_score.item(),
            "sensitivity_score": sensitivity_score.item(),
            "specificity_score": specificity_score.item(),
            "accuracy_score": accuracy_score.item(),
            "shape": input_size,
        }

    if os.path.exists(csv_output_path):
        os.remove(csv_output_path)

    with open(csv_output_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        # 如果是新文件，添加标题行
        writer.writerow(
            [
                "FileName",
                "Shape",
                "Jaccard",
                "Dice",
                "Sensitivity",
                "Specificity",
                "Accuracy",
            ]
        )

    file_writer = open(csv_output_path, mode="a", newline="")
    csv_writer = csv.writer(file_writer)
    print(f"Saving metrics to {csv_output_path}")
    for seg_file in tqdm(seg_files):
        if seg_file.split(".")[-1] != "jpg":
            continue
        if not os.path.exists(os.path.join(predict_output, seg_file)):
            print(f"Predicted mask {seg_file} does not exist!")
            exit(-1)
        predicted_path = os.path.join(predict_output, seg_file)
        if not os.path.exists(os.path.join(true_mask_path, seg_file)):
            seg_file = seg_file.split(".")[0] + "_segmentation.png"
            if not os.path.exists(os.path.join(true_mask_path, seg_file)):
                print(f"Ground truth mask {seg_file} does not exist!")
                exit(-1)
        groud_truth_path = os.path.join(true_mask_path, seg_file)

        metrics_dict = calculate_metrics_from_path(predicted_path, groud_truth_path)

        record = (
            [seg_file]
            + [str(metrics_dict["shape"][0]) + "x" + str(metrics_dict["shape"][1])]
            + [metrics_dict["jaccard_score"]]
            + [metrics_dict["dice_score"]]
            + [metrics_dict["sensitivity_score"]]
            + [metrics_dict["specificity_score"]]
            + [metrics_dict["accuracy_score"]]
        )
        csv_writer.writerow(record)
