import os
import cv2
import csv
from librosa import ex
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib import colors
from tqdm import tqdm

home_directory = os.path.expanduser("~")
from matplotlib.patches import RegularPolygon, Patch
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument(
    "--predict-input", type=str, default=None, help="Predict input path"
)
parser.add_argument(
    "--intergrated-output", type=str, default=None, help="Intergrated output path"
)
parser.add_argument("--image", type=str, default=None, help="original image path")
parser.add_argument("--points-csv", type=str, default=None, help="points csv path")
parser.add_argument("--color", type=str, default=None, help="color")
parser.add_argument("--revised", type=bool, help="revised version", default=False)

args = parser.parse_args()


root_path = args.predict_input
original_image_root_path = args.image
output_path = args.intergrated_output

if not os.path.exists(output_path):
    os.makedirs(output_path)

point_csv_path = args.points_csv


def save_legend(output_dir):
    """Save legend as a separate figure"""
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis("off")
    
    # Create legend handles
    handles = []
    
    # Input points (bright green)
    handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="black",
            markerfacecolor="#00ff00",
            markersize=15,
            label="Input points",
            linestyle="None",
            markeredgewidth=1,
        )
    )
    
    # Positive supplement points (bright yellow)
    handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="black",
            markerfacecolor="#ffff00",
            markersize=15,
            label="Positive supplement point",
            linestyle="None",
            markeredgewidth=1,
        )
    )
    
    # Negative supplement points (bright blue)
    handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="black",
            markerfacecolor="#00bfff",
            markersize=15,
            label="Negative supplement point",
            linestyle="None",
            markeredgewidth=1,
        )
    )
    
    # Ground Truth line (green)
    handles.append(
        Line2D(
            [0],
            [0],
            color="green",
            linewidth=2,
            label="Ground Truth",
        )
    )
    
    # PSAM output line (red)
    handles.append(
        Line2D(
            [0],
            [0],
            color="red",
            linewidth=2,
            label="PSAM output",
        )
    )
    
    # Create legend without frame and shadow
    legend = ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        ncol=5,
        columnspacing=1.5,
        handletextpad=0.5,
        fontsize=11,
    )
    
    # Save legend figure
    legend_path = os.path.join(output_dir, "legend.png")
    plt.savefig(legend_path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Legend saved to: {legend_path}")


# Save legend once before processing images
save_legend(output_path)

file_names = []
mask_width, mask_height = 1024, 1024  # final_mask.shape
if args.revised:
    file_names = [
        "ISIC_0024985",
        "ISIC_0024317",
        "ISIC_0027571",
        "ISIC_0029804",
        "ISIC_0029334",
        "ISIC_0026745",
    ]
else:
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(original_image_root_path):
        # 获取文件的完整路径
        file_path = os.path.join(original_image_root_path, file_name)
        # 如果是文件而不是文件夹
        if os.path.isfile(file_path):
            # 去除文件扩展名并将文件名添加到列表中
            if file_path.endswith(".jpg"):
                file_names.append(os.path.splitext(file_name)[0])


for file_name in tqdm(file_names):
    fig, ax = plt.subplots()
    ax.set_xlim(0, mask_width)
    ax.set_ylim(mask_height, 0)

    # 加载原始图像
    original_image_path = os.path.join(original_image_root_path, file_name + ".jpg")
    original_img = cv2.imread(original_image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img = cv2.resize(original_img, (mask_width, mask_height))
    # print("original_img", original_img.shape)

    if not os.path.exists(os.path.join(root_path, file_name + ".jpg")):
        print(f"{file_name}.jpg not found in {root_path}")
        exit(-1)
        continue
    # 加载mask图像
    final_mask_path = os.path.join(root_path, file_name + ".jpg")
    final_mask = cv2.imread(final_mask_path, cv2.IMREAD_GRAYSCALE)
    final_mask = cv2.resize(final_mask, (mask_width, mask_height))
    # print("final_mask", final_mask.shape)

    # coarse_mask_path = os.path.join(root_path, file_name + "_coarse.png")
    coarse_mask_path = os.path.join(root_path, file_name + "_refined.png")
    coarse_mask = cv2.imread(coarse_mask_path, cv2.IMREAD_GRAYSCALE)
    # print("coarse_mask", coarse_mask.shape)

    # refined_mask_path = os.path.join(root_path, file_name + "_refined.png")
    refined_mask_path = os.path.join(root_path, file_name + "_coarse.png")
    refined_mask = cv2.imread(refined_mask_path, cv2.IMREAD_GRAYSCALE)
    # print("refined_mask", refined_mask.shape)

    uncertain_mask_path = os.path.join(root_path, file_name + "_uncertain.png")
    # 读取灰度图
    gray_image = cv2.imread(uncertain_mask_path, cv2.IMREAD_GRAYSCALE)

    # 创建图形和坐标轴
    fig, ax = plt.subplots()
    ax.set_xlim(0, gray_image.shape[1])
    ax.set_ylim(gray_image.shape[0], 0)

    # 使用 Normalize 将灰度图像标准化到 [0, 1]
    # norm = colors.Normalize(vmin=0, vmax=1)
    normalized_image = gray_image / 255.0

    #
    # 1.	twilight: 一种柔和的渐变配色，适合表现细腻的渐变。
    # 2.	magma: 具有暖色调的渐变，非常适合视觉舒适的颜色过渡。
    # 3.	cubehelix: 这种配色保证了线性亮度的变化，非常适合数据可视化。
    # 显示彩色图像，使用 'jet' colormap 或其他你喜欢的 colormap
    cax = ax.imshow(normalized_image, cmap=args.color)

    # 添加颜色条
    # fig.colorbar(cax)
    fig.colorbar(cax, ax=ax)
    # ax.imshow(original_img)
    ax.axis("off")

    # # 查找每个mask的轮廓
    # final_contours, _ = cv2.findContours(
    #     final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    # )
    # coarse_contours, _ = cv2.findContours(
    #     coarse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    # )
    # refined_contours, _ = cv2.findContours(
    #     refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    # )

    # def find_top(contours):
    #     top_value = -1
    #     for contour in contours:
    #         contour = contour.squeeze(1)
    #         if contour.shape[0] > top_value:
    #             top_value = contour.shape[0]
    #     return top_value

    # def show_contour(contours, c, l, top_value):
    #     for contour in contours:
    #         contour = contour.squeeze(1)
    #         if contour.shape[0] < top_value:
    #             continue
    #         plt.plot(contour[:, 0], contour[:, 1], color=c, linewidth=l)

    # show_contour(final_contours, "r", 2, find_top(final_contours))
    # show_contour(coarse_contours, "g", 1, find_top(coarse_contours))
    # show_contour(refined_contours, "b", 1, find_top(refined_contours))

    p_point_mask_path = os.path.join(root_path, file_name + "_ps.png")
    p_point_mask = cv2.imread(p_point_mask_path, 0)
    n_point_mask_path = os.path.join(root_path, file_name + "_ns.png")
    n_point_mask = cv2.imread(n_point_mask_path, 0)

    # # Compute the difference between refined_mask and coarse_mask
    # difference_mask = cv2.absdiff(refined_mask, coarse_mask)
    # 计算 refined_mask 和 coarse_mask 的差集区域
    # 差集区域是：refined_mask 不为 0，且 coarse_mask 为 0 的区域
    # difference_mask = np.logical_and(refined_mask == 0, coarse_mask != 0)

    # 遍历 p_point_mask 中的像素，标记在 original_img 上
    check = True

    points = cv2.findNonZero(p_point_mask)
    if points is not None:
        for point in points:
            x, y = point[0]
            # 计算调整后的坐标
            adjusted_x = int(x * original_img.shape[1] / p_point_mask.shape[1])
            adjusted_y = int(y * original_img.shape[0] / p_point_mask.shape[0])

            # Draw star marker with high contrast
            plt.scatter(
                adjusted_x,
                adjusted_y,
                marker="*",
                s=500,
                color="#ffff00",
                edgecolors="black",
                linewidth=1,
                zorder=10,
            )

    points = cv2.findNonZero(n_point_mask)
    if points is not None:
        for point in points:
            x, y = point[0]
            # 计算调整后的坐标
            adjusted_x = int(x * original_img.shape[1] / n_point_mask.shape[1])
            adjusted_y = int(y * original_img.shape[0] / n_point_mask.shape[0])
            # print(adjusted_x, adjusted_y)
            # Draw star marker with high contrast
            plt.scatter(
                adjusted_x,
                adjusted_y,
                marker="*",
                s=500,
                color="#00bfff",
                edgecolors="black",
                linewidth=1,
                zorder=10,
            )

    # 显示原始点
    def find_coordinates_by_filename(csv_path, image_file_name):
        # 用于存储匹配的坐标
        coordinates_list = []
        labels = []
        shape = []
        # 打开CSV文件并搜索匹配的行
        with open(csv_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row["img_filename"] == image_file_name:
                    shape = [int(1024), int(1024)]
                    # 将每个坐标点添加到列表中
                    coordinates = []
                    for i in range(3):  # 假设有10个坐标点
                        x = float(row[f"x_{i}"])
                        y = float(row[f"y_{i}"])
                        coordinates.append([y, x])
                        labels.append(1)
                    coordinates_list.append(coordinates)
                    return coordinates_list[0], labels, shape

        return coordinates_list[0], labels, shape

    def scale_coordinates(coordinates, original_shape, target_shape):
        # 计算尺寸比例
        ratio_x = target_shape[1] / original_shape[1]
        ratio_y = target_shape[0] / original_shape[0]

        # 缩放坐标点
        scaled_coordinates = []
        for coord in coordinates:
            x_scaled = int(coord[0] * ratio_x)
            y_scaled = int(coord[1] * ratio_y)
            scaled_coordinates.append([x_scaled, y_scaled])
        return scaled_coordinates

    coord_ponts, coord_lables, shape = find_coordinates_by_filename(
        point_csv_path, file_name + ".jpg"
    )
    scaled_coordinates = scale_coordinates(coord_ponts, shape, original_img.shape)
    for sx, sy in scaled_coordinates:
        plt.scatter(
            sx, sy, marker="*", s=500, color="#00ff00", edgecolors="black", linewidth=1, zorder=10
        )
    # # 定义图例
    # handles, labels = ax.get_legend_handles_labels()
    # custom_legend = Line2D(
    #     [0],
    #     [0],
    #     marker="*",
    #     color="w",
    #     markerfacecolor="green",
    #     markersize=10,
    #     label="input points",
    #     linewidth=0.5,
    # )
    # handles.append(custom_legend)
    # custom_legend = Line2D(
    #     [0],
    #     [0],
    #     marker="*",
    #     color="w",
    #     markerfacecolor="#dc307e",
    #     markersize=10,
    #     label="positive points",
    #     linewidth=0.5,
    # )
    # handles.append(custom_legend)
    # custom_legend = Line2D(
    #     [0],
    #     [0],
    #     marker="*",
    #     color="w",
    #     markerfacecolor="#1c1cf5",
    #     markersize=10,
    #     label="nagative points",
    #     linewidth=0.5,
    # )
    # handles.append(custom_legend)
    # handles.append(Line2D([0], [0], color="green", lw=1, label="sparse mask"))
    # handles.append(Line2D([0], [0], color="blue", lw=1, label="dense mask"))
    # handles.append(Line2D([0], [0], color="red", lw=2, label="output mask"))
    # ax.legend(handles=handles, loc='upper right')
    # ax.legend(handles=handles, bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)

    plt.savefig(
        os.path.join(output_path, file_name + "_" + args.color + "_intergration.jpg"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)
    print(os.path.join(output_path, file_name + "_" + args.color + "_intergration.jpg"))

# Common:
# source /home/share/anaconda/bin/activate
# conda activate jiajun
# cd ~/zu52/STI/
# python 27.1.sp_den_mask_integration_V2.py
