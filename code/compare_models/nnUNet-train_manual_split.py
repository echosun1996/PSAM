import argparse
import os
import sys
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
from rich import print


parser = argparse.ArgumentParser(description="Run nnUNet to predict masks")

parser.add_argument("--json-path", type=str, help="json path")

parser.add_argument("--training-image-count", type=int, help="training image count ")
parser.add_argument("--val-image-count", type=int, help="val image count ")

args = parser.parse_args()

splits = []
for fold in range(1):
    splits.append(
        {
            "train": ["img-" + str(i) for i in range(1, args.training_image_count + 1)],
            "val": [
                "img-" + str(i)
                for i in range(
                    args.training_image_count + 1,
                    args.training_image_count + args.val_image_count + 1,
                )
            ],
        }
    )
print("Current split:", len(splits))
save_json(splits, args.json_path)
