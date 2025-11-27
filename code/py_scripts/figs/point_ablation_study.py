import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from matplotlib.patches import RegularPolygon
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument(
    "--predict-input", type=str, default=None, help="Predict input path (mask directory)"
)
parser.add_argument(
    "--intergrated-output", type=str, default=None, help="Integrated output path"
)
parser.add_argument("--image", type=str, default=None, help="original image path")
parser.add_argument("--gt-mask", type=str, default=None, help="ground truth mask path")
parser.add_argument("--points-csv", type=str, default=None, help="points csv path")
parser.add_argument("--single-image", type=str, default=None, help="Process only a single image (e.g., ISIC_0024985.jpg). If None, process all images.")
parser.add_argument("--prompt-type", type=str, default="P_B", help="Prompt type: P_B, P, or B. B mode will not show green input points.")

args = parser.parse_args()

root_path = args.predict_input
original_image_root_path = args.image
gt_mask_path = args.gt_mask
output_path = args.intergrated_output

if not os.path.exists(output_path):
    os.makedirs(output_path)

point_csv_path = args.points_csv


def save_legend(output_dir, prompt_type="P_B"):
    """Save legend as a separate figure"""
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis("off")
    
    # Create legend handles
    handles = []
    
    # Input points (bright green) - only show if prompt_type is not "B"
    if prompt_type != "B":
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
                markeredgewidth=1.5,
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
            markeredgewidth=1.5,
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
            markeredgewidth=1.5,
        )
    )
    
    # Ground Truth line (green)
    handles.append(
        Line2D(
            [0],
            [0],
            color="green",
            linewidth=3,
            label="Ground Truth",
        )
    )
    
    # Prediction line (red)
    handles.append(
        Line2D(
            [0],
            [0],
            color="red",
            linewidth=3,
            label="Prediction",
        )
    )
    
    # Create legend without frame and shadow
    ncol = len(handles)
    legend = ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        ncol=ncol,
        columnspacing=1.5,
        handletextpad=0.5,
        fontsize=11,
    )
    
    # Save legend figure
    legend_path = os.path.join(output_dir, "legend.png")
    plt.savefig(legend_path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Legend saved to: {legend_path}")


def find_coordinates_by_filename(csv_path, image_file_name):
    """Find coordinates from CSV file by image filename"""
    coordinates_list = []
    labels = []
    shape = []
    with open(csv_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["img_filename"] == image_file_name:
                shape = [int(1024), int(1024)]
                coordinates = []
                for i in range(3):  # Assuming 3 coordinate points
                    try:
                        x = float(row[f"x_{i}"])
                        y = float(row[f"y_{i}"])
                        coordinates.append([y, x])
                        labels.append(1)
                    except (KeyError, ValueError):
                        break
                coordinates_list.append(coordinates)
                return coordinates_list[0], labels, shape
    return coordinates_list[0] if coordinates_list else [], labels, shape


def scale_coordinates(coordinates, original_shape, target_shape):
    """Scale coordinates from original image size to target size"""
    ratio_x = target_shape[1] / original_shape[1]
    ratio_y = target_shape[0] / original_shape[0]
    
    scaled_coordinates = []
    for coord in coordinates:
        x_scaled = int(coord[0] * ratio_x)
        y_scaled = int(coord[1] * ratio_y)
        scaled_coordinates.append([x_scaled, y_scaled])
    return scaled_coordinates


# Save legend once before processing images
save_legend(output_path, args.prompt_type)

# Get list of image files
file_names = []
mask_width, mask_height = 1024, 1024

# If single-image is specified, only process that image
if args.single_image is not None:
    single_image_basename = os.path.splitext(args.single_image)[0]
    file_names.append(single_image_basename)
else:
    # Process all images in the directory
    for file_name in os.listdir(original_image_root_path):
        file_path = os.path.join(original_image_root_path, file_name)
        if os.path.isfile(file_path) and file_path.endswith(".jpg"):
            file_names.append(os.path.splitext(file_name)[0])


for file_name in tqdm(file_names):
    fig, ax = plt.subplots()
    ax.set_xlim(0, mask_width)
    ax.set_ylim(mask_height, 0)
    
    # Load original image
    original_image_path = os.path.join(original_image_root_path, file_name + ".jpg")
    if not os.path.exists(original_image_path):
        print(f"Warning: {original_image_path} not found, skipping...")
        continue
    
    original_img = cv2.imread(original_image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img = cv2.resize(original_img, (mask_width, mask_height))
    
    # Display original image as background
    ax.imshow(original_img)
    ax.axis("off")
    
    # Load prediction mask
    pred_mask_path = os.path.join(root_path, file_name + ".jpg")
    if not os.path.exists(pred_mask_path):
        print(f"Warning: {pred_mask_path} not found, skipping...")
        continue
    
    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
    pred_mask = cv2.resize(pred_mask, (mask_width, mask_height))
    
    # Load ground truth mask
    if gt_mask_path:
        # If gt_mask_path is a directory, construct path
        if os.path.isdir(gt_mask_path):
            gt_mask_file = os.path.join(gt_mask_path, file_name + "_segmentation.png")
        else:
            # If it's a file path, use it directly
            gt_mask_file = gt_mask_path
        
        if os.path.exists(gt_mask_file):
            gt_mask = cv2.imread(gt_mask_file, cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                gt_mask = cv2.resize(gt_mask, (mask_width, mask_height))
                
                # Draw ground truth contours in green
                gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in gt_contours:
                    contour = contour.squeeze(1)
                    if contour.shape[0] > 0:
                        ax.plot(contour[:, 0], contour[:, 1], color='green', linewidth=3, label='Ground Truth')
    
    # Draw prediction mask contours in red
    if np.any(pred_mask > 0):
        pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in pred_contours:
            contour = contour.squeeze(1)
            if contour.shape[0] > 0:
                ax.plot(contour[:, 0], contour[:, 1], color='red', linewidth=3, label='Prediction')
    
    # Draw pink pentagrams (positive supplement points) - ALWAYS show
    ps_mask_path = os.path.join(root_path, file_name + "_ps.png")
    if os.path.exists(ps_mask_path):
        ps_mask = cv2.imread(ps_mask_path, cv2.IMREAD_GRAYSCALE)
        if ps_mask is not None:
            # Read original size, then scale coordinates
            ps_points = cv2.findNonZero(ps_mask)
            if ps_points is not None:
                ps_points = ps_points.squeeze(1)
                for point in ps_points:
                    x, y = point[0], point[1]
                    # Calculate adjusted coordinates based on original mask size
                    adjusted_x = int(x * original_img.shape[1] / ps_mask.shape[1])
                    adjusted_y = int(y * original_img.shape[0] / ps_mask.shape[0])
                    ax.scatter(
                        adjusted_x,
                        adjusted_y,
                        marker="*",
                        s=500,
                        c="#ffff00",
                        edgecolors="black",
                        linewidths=1.5,
                        zorder=10,
                    )
    
    # Draw blue pentagrams (negative supplement points) - ALWAYS show
    ns_mask_path = os.path.join(root_path, file_name + "_ns.png")
    if os.path.exists(ns_mask_path):
        ns_mask = cv2.imread(ns_mask_path, cv2.IMREAD_GRAYSCALE)
        if ns_mask is not None:
            # Read original size, then scale coordinates
            ns_points = cv2.findNonZero(ns_mask)
            if ns_points is not None:
                ns_points = ns_points.squeeze(1)
                for point in ns_points:
                    x, y = point[0], point[1]
                    # Calculate adjusted coordinates based on original mask size
                    adjusted_x = int(x * original_img.shape[1] / ns_mask.shape[1])
                    adjusted_y = int(y * original_img.shape[0] / ns_mask.shape[0])
                    ax.scatter(
                        adjusted_x,
                        adjusted_y,
                        marker="*",
                        s=500,
                        c="#00bfff",
                        edgecolors="black",
                        linewidths=1.5,
                        zorder=10,
                    )
    
    # Draw green pentagrams (input points from CSV)
    # Only show green points if prompt_type is not "B"
    # P_B and P modes: show green, pink, blue
    # B mode: show only pink and blue (no green)
    if args.prompt_type != "B" and point_csv_path and os.path.exists(point_csv_path):
        coord_points, coord_labels, shape = find_coordinates_by_filename(
            point_csv_path, file_name + ".jpg"
        )
        if coord_points:
            scaled_coordinates = scale_coordinates(coord_points, shape, original_img.shape)
            for sx, sy in scaled_coordinates:
                ax.scatter(
                    sx, sy, marker="*", s=500, c="#00ff00", edgecolors="black", linewidths=1.5, zorder=10
                )
    
    # Save figure
    output_filename = file_name + "_combined.jpg"
    plt.savefig(
        os.path.join(output_path, output_filename),
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )
    plt.close(fig)
    print(f"Saved: {os.path.join(output_path, output_filename)}")

