import os

import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--input", "-i", type=str, help="input path")
parser.add_argument("--output", "-o", type=str, help="output path")

args = parser.parse_args()


def save_file_names_to_txt(directory, output_path, folder_name):
    # 确保目录存在且是一个目录
    if not os.path.isdir(directory):
        print("指定的路径不是一个有效的目录。")
        return
    # 获取目录下所有文件名
    files = os.listdir(directory)

    # 筛选出文件名（去除目录和子目录）
    file_names = [
        file for file in files if os.path.isfile(os.path.join(directory, file))
    ]

    # 将文件名写入txt文件
    with open(os.path.join(output_path), "w") as f:
        for file_name in file_names:
            file_name = file_name.split(".")[0]
            f.write(folder_name + "/" + file_name + "\n")

    print("文件名已保存到 " + output_path + " 中。")


# 指定目录路径
# directory_path = "/home/jsun/zu52_scratch/STI/PSOD_Data/ISIC_2016_Test/ISBI2016_ISIC_Part1_Test_Data"
# output_path = "/home/jsun/zu52_scratch/STI/PSOD_Data/train.txt"
# save_file_names_to_txt(directory_path,output_path,"ISIC_2016_Test")

directory_path = args.input
output_path = args.output

# directory_path = "data/HAM10000/input/test/HAM10000_img"

# directory_path = "data/HAM10000/input/test/HAM10000_img"
sub_path = "/".join(directory_path.split("/")[-2:])

# print(sub_path)

# directory_path = "/home/jsun/zu52_scratch/STI/WSCOD/train/Image/HAM10000"
# output_path = "/home/jsun/zu52_scratch/STI/PSOD_Data/train.txt"
save_file_names_to_txt(directory_path, output_path, sub_path)


# mkdir -vp /home/jsun/zu52_scratch/STI/PSOD_Data/train/Image/
# cp -r HAM10000/HAM10000_images/ /home/jsun/zu52_scratch/STI/PSOD_Data/train/Image/HAM10000
