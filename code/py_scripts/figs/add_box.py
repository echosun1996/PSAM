import cv2
import os
import argparse
import numpy as np


# 提供的函数，生成 bounding box
def generate_box_from_gt(gt_path, expand_ratio=0.25):
    # 读取分割图像
    segmentation_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    # 找到图像中的非零像素（表示目标的区域）
    coords = np.column_stack(np.where(segmentation_img > 0))

    # 如果图像中没有目标，返回None
    if len(coords) == 0:
        return None

    # 计算bounding box
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    # 获取图像的高度和宽度
    img_height, img_width = segmentation_img.shape

    # 计算box的宽度和高度
    box_width = x_max - x_min
    box_height = y_max - y_min

    # 计算扩展后的宽度和高度
    expand_width = box_width * expand_ratio
    expand_height = box_height * expand_ratio

    # 增加bounding box的范围，并确保不会超出图像的边界
    x_min = max(0, int(x_min - expand_width // 2))
    y_min = max(0, int(y_min - expand_height // 2))
    x_max = min(img_width - 1, int(x_max + expand_width // 2))
    y_max = min(img_height - 1, int(y_max + expand_height // 2))

    # 返回增大的bounding box
    return [x_min, y_min, x_max, y_max]


def draw_box_on_image(source_file, box):
    # 读取源文件的图像
    image = cv2.imread(source_file)

    # 如果没有找到box，返回原图像
    if box is None:
        print(f"No bounding box found for the provided mask.")
        return image

    # 提取box的坐标 [y_min, x_min, y_max, x_max]
    y_min, x_min, y_max, x_max = box

    # 将颜色 #61efee 转换为 BGR 格式 (238, 239, 97)
    box_color = (238, 239, 97)

    # 绘制矩形框，使用 box_color，线条粗细为2
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), box_color, 2)

    return image


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="Generate bounding box and add it to the source image."
    )
    parser.add_argument(
        "--source-file", type=str, required=True, help="Path to the source image file."
    )
    parser.add_argument(
        "--gt-mask-path", type=str, required=True, help="Path to the ground truth mask."
    )

    args = parser.parse_args()

    # 生成bounding box
    box = generate_box_from_gt(args.gt_mask_path, expand_ratio=0.4)

    # 在源图像上绘制bounding box
    image_with_box = draw_box_on_image(args.source_file, box)

    # 将带有bounding box的图像保存，覆盖原图像
    cv2.imwrite(args.source_file, image_with_box)
    print(f"Bounding box added and image saved at {args.source_file}")


if __name__ == "__main__":
    main()
