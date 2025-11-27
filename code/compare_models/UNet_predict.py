import subprocess
import os
from cv2 import RETR_FLOODFILL
from sympy import im
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description="Run UNet to predict masks")

parser.add_argument("--input", "-i", type=str, help="input image path")
parser.add_argument("--output", "-o", type=str, help="output mask path")

args = parser.parse_args()

image_folder_path = args.input
output_folder_path = args.output


# 要运行的命令
command = "python ./reps/UNet/predict.py --model ./checkpoints/UNet/checkpoint_epoch5.pth -s 1 -i"
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 获取image_folder_path下的全部文件名
file_names = os.listdir(image_folder_path)
batch_size = 1

for i in tqdm(range(0, len(file_names), batch_size)):
    batch_file_names = file_names[i : i + batch_size]
    images_path = ""
    outputs_path = ""
    for file_name in batch_file_names:
        image_path = os.path.join(image_folder_path, file_name)
        output_path = os.path.join(output_folder_path, file_name)

        images_path = images_path + " " + image_path
        outputs_path = outputs_path + " " + output_path

    if images_path == "":
        continue
    command = command + " " + images_path + " -o " + outputs_path
    subprocess.call(
        [
            "python ./reps/UNet/predict.py"
            + " --model"
            + " ./checkpoints/UNet/checkpoint_epoch5.pth"
            + " -s 1"
            + " -i "
            + images_path
            + " -o "
            + outputs_path,
        ],
        shell=True,
    )
