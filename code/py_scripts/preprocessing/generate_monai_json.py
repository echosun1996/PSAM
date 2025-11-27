from collections import OrderedDict
import json
import argparse
import os
from rich import print
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-name", type=str, default=None, help="dataset name")
parser.add_argument("--path", type=str, default=None, help="process path")
args = parser.parse_args()

json_dict = OrderedDict()
json_dict["name"] = args.dataset_name
json_dict["tensorImageSize"] = "2D"
root_path = args.path


def get_img_and_seg_folder_name(path):
    img_folder = None
    seg_folder = None
    for item in os.listdir(path):
        if os.path.isfile(item):
            continue
        if "img" in item:
            img_folder = item
            continue
        if "seg_binary" in item:
            seg_folder = item
            continue
    if img_folder is None or seg_folder is None:
        raise ValueError("img or seg folder not found")
    return img_folder, seg_folder


def path_process(folder_path, folder_type):
    folder_name = folder_path.split("/")[-1]
    if os.path.exists(folder_path):
        img_folder, seg_folder = get_img_and_seg_folder_name(folder_path)
        for item in tqdm(
            os.listdir(os.path.join(folder_path, img_folder)),
            desc=f"Processing {folder_type}...",
        ):
            if item.endswith(".jpg"):
                seg_path = os.path.join(
                    folder_path, seg_folder, item.replace(".jpg", "_segmentation.png")
                )
                if not os.path.exists(seg_path):
                    print(f"{seg_path} not found")
                    exit(-1)
                if folder_type not in json_dict:
                    json_dict[folder_type] = []
                json_dict[folder_type].append(
                    {
                        "image": os.path.join(folder_name, img_folder, item),
                        "label": os.path.join(
                            folder_name,
                            seg_folder,
                            item.replace(".jpg", "_segmentation.png"),
                        ),
                        "image_meta_dict": {"filename_or_obj": item},
                    }
                )


test_path = os.path.join(root_path, "test")
path_process(test_path, "test")
val_path = os.path.join(root_path, "val")
path_process(val_path, "validation")
test_path = os.path.join(root_path, "train")
path_process(test_path, "training")

# 保存json
with open(os.path.join(args.path, "dataset_0.json"), "w") as f:
    json.dump(json_dict, f, indent=4, separators=(",", ": "))
