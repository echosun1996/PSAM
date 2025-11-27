import argparse
from PIL import Image
import numpy as np
import os
from scipy.ndimage import binary_dilation


def add_scribbles_to_image(source_file, scribbles_path, is_positive):
    """
    在源图像上添加来自 scribble 图像的标注。

    :param source_file: 源图像的路径
    :param scribbles_path: scribble 图像的路径，白色部分代表 scribble
    :param is_positive: 是否为正面的涂鸦标注（正面标注为绿色，负面为红色）
    """
    # 加载源图像
    source_image = Image.open(source_file).convert("RGB")

    # 加载 scribble 图像（灰度图）
    scribble_image = Image.open(scribbles_path).convert("L")

    # 转换为 numpy 数组
    source_np = np.array(source_image)
    scribble_np = np.array(scribble_image)

    # 定义 scribble 的颜色（正面绿色，负面红色）
    if is_positive:
        scribble_color = [255, 255, 0]  # 黄色
    else:
        scribble_color = [128, 0, 128]  # 紫色

    # 将 scribble 的白色区域 (255) 转换为指定的颜色
    scribble_mask = scribble_np != 0  # 白色部分为 True

    # 加粗 Scribble 区域（使用膨胀操作）
    scribble_mask = binary_dilation(scribble_mask, iterations=3)  # 增加3像素宽度

    # 在源图像的相应位置添加 scribble
    source_np[scribble_mask] = scribble_color

    # 转换回图像并保存
    result_image = Image.fromarray(source_np)
    # save_path = source_file.replace(
    #     ".png", "_with_scribbles.png"
    # )  # 或者其他适当的保存路径
    result_image.save(source_file)
    print(f"Image saved with scribbles at {source_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在源图像上添加涂鸦")
    parser.add_argument("--source-file", type=str, required=True, help="源图像文件路径")
    parser.add_argument(
        "--scribbles-path",
        type=str,
        required=True,
        help="scribble 文件路径（图像格式）",
    )
    parser.add_argument(
        "--isPositive",
        type=bool,
        # required=True,
        default=False,
        help="是否是正面的涂鸦标注",
    )

    args = parser.parse_args()

    # 调用主函数
    add_scribbles_to_image(args.source_file, args.scribbles_path, args.isPositive)
