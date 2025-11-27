# 将HAM10000 转换到 ~/zu52_scratch/STI/HAM10000/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_cropped_data/Task666_HAM10000

import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Run nnUNet to predict masks")

parser.add_argument("--input", "-i", type=str, help="input path")
parser.add_argument("--data-path", type=str, help="output data folder path")

args = parser.parse_args()


out_root_path = os.path.join(
    args.data_path,
    "nnUNetFrame/DATASET/nnUNet_raw/nnUNet_cropped_data/Task666_HAM10000",
)

testing_input = os.path.join(out_root_path, "testing", "input")
testing_output = os.path.join(out_root_path, "testing", "output")
training_input = os.path.join(out_root_path, "training", "input")
training_output = os.path.join(out_root_path, "training", "output")

os.makedirs(testing_input, exist_ok=True)
os.makedirs(testing_output, exist_ok=True)
os.makedirs(training_input, exist_ok=True)
os.makedirs(training_output, exist_ok=True)

input_root = args.input

traing_img_path = os.path.join(input_root, "train/HAM10000_img")
traing_seg_path = os.path.join(input_root, "train/HAM10000_seg")

val_img_path = os.path.join(input_root, "val/HAM10000_img")
val_seg_path = os.path.join(input_root, "val/HAM10000_seg")

test_img_path = os.path.join(input_root, "test/HAM10000_img")
test_seg_path = os.path.join(input_root, "test/HAM10000_seg")

print("Start moving tringing data...")
traing_images_list = os.listdir(traing_img_path)
traing_images_list.sort()  # 对读取的图片进行排序
images_count = 0
for image in tqdm(traing_images_list):
    seg_image_name = image.split(".")[0] + "_segmentation.png"
    images_count += 1
    # 写入测试集
    file_name = "img-" + str(images_count) + ".png"
    os.system(
        "cp "
        + os.path.join(traing_img_path, image)
        + " "
        + os.path.join(training_input, file_name)
    )
    os.system(
        "cp "
        + os.path.join(traing_seg_path, seg_image_name)
        + " "
        + os.path.join(training_output, file_name)
    )

print("Start moving val data...")
val_images_list = os.listdir(val_img_path)
val_images_list.sort()  # 对读取的图片进行排序
for image in tqdm(val_images_list):
    seg_image_name = image.split(".")[0] + "_segmentation.png"
    images_count += 1
    # 写入测试集
    file_name = "img-" + str(images_count) + ".png"
    os.system(
        "cp "
        + os.path.join(val_img_path, image)
        + " "
        + os.path.join(training_input, file_name)
    )
    os.system(
        "cp "
        + os.path.join(val_seg_path, seg_image_name)
        + " "
        + os.path.join(training_output, file_name)
    )

print("Start moving test data...")
test_images_list = os.listdir(test_img_path)
test_images_list.sort()  # 对读取的图片进行排序
images_count = 0
for image in tqdm(test_images_list):
    seg_image_name = image.split(".")[0] + "_segmentation.png"
    images_count += 1
    # 写入测试集
    file_name = "img-" + str(images_count) + ".png"
    os.system(
        "cp "
        + os.path.join(test_img_path, image)
        + " "
        + os.path.join(testing_input, file_name)
    )
    os.system(
        "cp "
        + os.path.join(test_seg_path, seg_image_name)
        + " "
        + os.path.join(testing_output, file_name)
    )

print("nnUNet data move finish.")
