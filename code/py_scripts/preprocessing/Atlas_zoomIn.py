import sys
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input-img-folder", type=str, default=None, help="Atlas img folder path"
)
parser.add_argument(
    "--input-seg-folder", type=str, default=None, help="Atlas seg folder path"
)

parser.add_argument("--output-dir", default=None, type=str)
parser.add_argument("--padding", default=35, type=int)
args = parser.parse_args()


input_img_folder = args.input_img_folder
input_seg_folder = args.input_seg_folder
output_path = args.output_dir
padding = args.padding


output_img_path = os.path.join(
    output_path, "input", "test", "AtlasZoomIn" + str(padding) + "_img"
)
output_seg_path = os.path.join(
    output_path, "input", "test", "AtlasZoomIn" + str(padding) + "_seg"
)
os.makedirs(output_img_path, exist_ok=True)
os.makedirs(output_seg_path, exist_ok=True)

# 获取image_folder_path下的全部文件名
img_names = os.listdir(input_img_folder)
for img_name in tqdm(img_names):
    img_path = os.path.join(input_img_folder, img_name)
    seg_name = img_name.replace(".jpg", "_segmentation.png")
    seg_path = os.path.join(input_seg_folder, seg_name)

    out_img_path = os.path.join(output_img_path, img_name)
    out_seg_path = os.path.join(output_seg_path, seg_name)

    # 打开原始图像和分割图像
    im = Image.open(img_path)
    seg = Image.open(seg_path).convert("L")  # 将分割图像转为灰度图

    # 转换分割图像为numpy数组
    seg_np = np.array(seg)

    # 查找所有独立的mask区域
    contours, _ = cv2.findContours(seg_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果没有检测到轮廓，则跳过此图像
    if len(contours) == 0:
        print(f"No contour found in {img_name}, exit...")
        exit(1)

    # 找到轮廓
    contour = min(contours, key=cv2.contourArea)

    # 创建一个空白的mask，只保留contour的mask区域
    contour_mask = np.zeros_like(seg_np)
    cv2.drawContours(contour_mask, [contour], -1, color=255, thickness=cv2.FILLED)

    # 用mask更新原始分割图像，去除其他区域的mask
    seg_np = seg_np * (contour_mask // 255)

    # 将更新后的seg_np转换回PIL图像
    seg = Image.fromarray(seg_np)

    # 计算mask的边界框，并加上padding
    x, y, w, h = cv2.boundingRect(contour)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(seg_np.shape[1] - x, w + 2 * padding)
    h = min(seg_np.shape[0] - y, h + 2 * padding)

    # 提取原图中对应mask的部分
    cropped_im = im.crop((x, y, x + w, y + h))
    cropped_seg = seg.crop((x, y, x + w, y + h))

    # 将裁剪后的部分resize回1024x1024
    resized_im = cropped_im.resize((1024, 1024), Image.Resampling.LANCZOS)
    resized_seg = cropped_seg.resize((1024, 1024), Image.Resampling.NEAREST)

    # 保存结果图像和mask
    resized_im.save(out_img_path)
    resized_seg.save(out_seg_path)
