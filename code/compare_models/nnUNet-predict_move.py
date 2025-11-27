import csv
import os
from tqdm import tqdm
from skimage import io
import argparse

parser = argparse.ArgumentParser(description="Run nnUNet to predict masks")

parser.add_argument("--input-img-path", type=str, help="input image path")
parser.add_argument("--input-seg-path", type=str, help="input seg path")
parser.add_argument("--output-path", type=str, help="output data folder path")

args = parser.parse_args()

input_path = os.path.join(args.input_img_path)
seg_path = os.path.join(args.input_seg_path)
out_root_path = os.path.join(args.output_path)

# 已知的图片名称和对应的文件名字典
image_file_mapping = {}


# 假设这是一个从已有的CSV文件中读取已有映射关系的函数
def load_existing_mapping(csv_file_name):
    with open(csv_file_name, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # 跳过列名行
        for row in reader:
            image_file_mapping[row[0]] = row[1]


# 假设这是一个函数，用于检查image_name是否存在，以及对应的file_name是否一致
def check_image_file_exist(image_name, file_name):
    if image_name in image_file_mapping:
        existing_file_name = image_file_mapping[image_name]
        if existing_file_name == file_name:
            return True
        else:
            print(
                f"Image name '{image_name}' already exists, but file name '{file_name}' is different (existing file name: '{existing_file_name}')."
            )
            exit(-1)
    else:
        return False


testing_input = os.path.join(out_root_path, "input")
testing_output = os.path.join(out_root_path, "output")
os.makedirs(testing_input, exist_ok=True)
os.makedirs(testing_output, exist_ok=True)

# record the file map.
csv_file_path = os.path.join(out_root_path, "image_file_mapping.csv")

if not os.path.exists(csv_file_path):
    file = open(csv_file_path, mode="a", newline="")
    writer = csv.writer(file)
    writer.writerow(["Original Image Name", "Covert File Name"])
else:
    load_existing_mapping(csv_file_path)
    file = open(csv_file_path, mode="a", newline="")
    writer = csv.writer(file)

images_list = os.listdir(input_path)
images_list.sort()  # 对读取的图片进行排序

images_sum = len(images_list)
test_count = 0
for image in tqdm(images_list):
    seg_image_name = image.split(".")[0] + "_segmentation.png"
    if not os.path.exists(os.path.join(seg_path, seg_image_name)):
        seg_image_name = image.split(".")[0] + "_Segmentation.png"

    test_count += 1

    file_name = "img-" + str(test_count) + ".png"
    if not check_image_file_exist(image, file_name):
        writer.writerow([image, file_name])
    os.system(
        "cp "
        + os.path.join(input_path, image)
        + " "
        + os.path.join(testing_input, file_name)
    )

    os.system(
        "cp "
        + os.path.join(seg_path, seg_image_name)
        + " "
        + os.path.join(testing_output, file_name)
    )

print("images_sum: ", images_sum)
