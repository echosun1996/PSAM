import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib.patches import RegularPolygon
from matplotlib.lines import Line2D

def find_coordinates_by_filename(csv_path, image_file_name):
    """Find coordinates from CSV file by image filename"""
    coordinates = []
    labels = []
    with open(csv_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["img_filename"] == image_file_name:
                # Try different CSV formats
                # Format 1: x_0, y_0, x_1, y_1, x_2, y_2 (3 points)
                if "x_0" in row:
                    for i in range(3):
                        if f"x_{i}" in row and f"y_{i}" in row:
                            try:
                                x = float(row[f"x_{i}"])
                                y = float(row[f"y_{i}"])
                                coordinates.append((int(x), int(y)))
                                labels.append(1)  # Positive points
                            except (ValueError, KeyError):
                                pass
                # Format 2: x_i, y_i (single point)
                elif "x_i" in row and "y_i" in row:
                    try:
                        x = float(row["x_i"])
                        y = float(row["y_i"])
                        coordinates.append((int(x), int(y)))
                        labels.append(1)
                    except (ValueError, KeyError):
                        pass
                break  # Found the matching row, exit loop
    return coordinates, labels

def scale_coordinates(coords, original_shape, target_shape):
    """Scale coordinates from original image size to target size"""
    scale_x = target_shape[1] / original_shape[1]
    scale_y = target_shape[0] / original_shape[0]
    scaled_coords = [(int(x * scale_x), int(y * scale_y)) for x, y in coords]
    return scaled_coords

def combine_gt_pred_pentagrams(original_img_path, pred_mask_path, gt_mask_path, 
                                ps_mask_path, ns_mask_path, points_csv_path,
                                output_path, image_filename):
    """
    Combine ground truth mask, prediction mask, and three-colored pentagrams
    """
    # Load images
    original_img = cv2.imread(original_img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize masks to match original image size
    target_shape = original_img.shape[:2]
    pred_mask = cv2.resize(pred_mask, (target_shape[1], target_shape[0]))
    gt_mask = cv2.resize(gt_mask, (target_shape[1], target_shape[0]))
    
    # Load pentagram masks
    ps_mask = cv2.imread(ps_mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(ps_mask_path) else None
    ns_mask = cv2.imread(ns_mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(ns_mask_path) else None
    
    if ps_mask is not None:
        ps_mask = cv2.resize(ps_mask, (target_shape[1], target_shape[0]))
    if ns_mask is not None:
        ns_mask = cv2.resize(ns_mask, (target_shape[1], target_shape[0]))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, target_shape[1])
    ax.set_ylim(target_shape[0], 0)
    
    # Display original image
    ax.imshow(original_img)
    ax.axis("off")
    
    # Draw ground truth contours in green
    gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in gt_contours:
        contour = contour.squeeze(1)
        if contour.shape[0] > 0:
            ax.plot(contour[:, 0], contour[:, 1], color='green', linewidth=3, label='Ground Truth')
    
    # Draw prediction contours in red
    if np.any(pred_mask > 0):
        pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in pred_contours:
            contour = contour.squeeze(1)
            if contour.shape[0] > 0:
                ax.plot(contour[:, 0], contour[:, 1], color='red', linewidth=3, label='Prediction')
    
    # Draw pink pentagrams (positive supplement points)
    if ps_mask is not None:
        ps_points = cv2.findNonZero(ps_mask)
        if ps_points is not None:
            ps_points = ps_points.squeeze(1)
            for point in ps_points:
                x, y = point[0], point[1]
                pentagram = RegularPolygon((x, y), numVertices=5, radius=8, 
                                          orientation=np.pi/2, color='magenta', 
                                          edgecolor='white', linewidth=1)
                ax.add_patch(pentagram)
    
    # Draw blue pentagrams (negative supplement points)
    if ns_mask is not None:
        ns_points = cv2.findNonZero(ns_mask)
        if ns_points is not None:
            ns_points = ns_points.squeeze(1)
            for point in ns_points:
                x, y = point[0], point[1]
                pentagram = RegularPolygon((x, y), numVertices=5, radius=8, 
                                          orientation=np.pi/2, color='blue', 
                                          edgecolor='white', linewidth=1)
                ax.add_patch(pentagram)
    
    # Draw green pentagrams (input points from CSV)
    if points_csv_path and os.path.exists(points_csv_path):
        # CSV coordinates are typically in 1024x1024 format
        csv_shape = (1024, 1024)
        coords, labels = find_coordinates_by_filename(points_csv_path, image_filename)
        scaled_coords = scale_coordinates(coords, csv_shape, target_shape)
        
        for x, y in scaled_coords:
            pentagram = RegularPolygon((x, y), numVertices=5, radius=8, 
                                      orientation=np.pi/2, color='green', 
                                      edgecolor='white', linewidth=1)
            ax.add_patch(pentagram)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined image saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine ground truth mask, prediction mask, and three-colored pentagrams"
    )
    parser.add_argument("--original-img", type=str, required=True, help="Path to original image")
    parser.add_argument("--pred-mask", type=str, required=True, help="Path to prediction mask")
    parser.add_argument("--gt-mask", type=str, required=True, help="Path to ground truth mask")
    parser.add_argument("--ps-mask", type=str, required=True, help="Path to positive supplement points mask (_ps.png)")
    parser.add_argument("--ns-mask", type=str, required=True, help="Path to negative supplement points mask (_ns.png)")
    parser.add_argument("--points-csv", type=str, required=True, help="Path to points CSV file")
    parser.add_argument("--output", type=str, required=True, help="Output path for combined image")
    parser.add_argument("--image-filename", type=str, required=True, help="Image filename (e.g., ISIC_0024985.jpg)")
    
    args = parser.parse_args()
    
    combine_gt_pred_pentagrams(
        args.original_img,
        args.pred_mask,
        args.gt_mask,
        args.ps_mask,
        args.ns_mask,
        args.points_csv,
        args.output,
        args.image_filename
    )

