#!/usr/bin/env python3
"""
Script to add pentagram markers to visualization images.
This script adds green pentagrams for input points from CSV file.
"""

import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.lines import Line2D


def find_coordinates_by_filename(csv_path, image_file_name):
    """Find coordinates from CSV file for a specific image."""
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
    """Scale coordinates from original shape to target shape."""
    ratio_x = target_shape[1] / original_shape[1]
    ratio_y = target_shape[0] / original_shape[0]
    
    scaled_coordinates = []
    for coord in coordinates:
        x_scaled = int(coord[0] * ratio_x)
        y_scaled = int(coord[1] * ratio_y)
        scaled_coordinates.append([x_scaled, y_scaled])
    return scaled_coordinates


def add_pentagrams_to_visualization(
    vis_image_path,
    original_image_path,
    csv_path,
    output_path,
    image_filename
):
    """
    Add pentagram markers to a visualization image.
    
    Args:
        vis_image_path: Path to the existing visualization image
        original_image_path: Path to the original image (for size reference)
        csv_path: Path to CSV file with point coordinates
        output_path: Path to save the output image
        image_filename: Name of the image file
    """
    # Load the visualization image
    vis_img = cv2.imread(vis_image_path)
    if vis_img is None:
        print(f"Error: Could not load visualization image: {vis_image_path}")
        return False
    
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    
    # Load original image for size reference
    original_img = cv2.imread(original_image_path)
    if original_img is None:
        print(f"Error: Could not load original image: {original_image_path}")
        return False
    
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img = cv2.resize(original_img, (vis_img.shape[1], vis_img.shape[0]))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(vis_img)
    ax.axis('off')
    
    # Get coordinates from CSV
    coord_points, coord_labels, shape = find_coordinates_by_filename(
        csv_path, image_filename
    )
    
    if coord_points:
        # Scale coordinates to match visualization image size
        scaled_coordinates = scale_coordinates(
            coord_points, 
            original_img.shape[:2], 
            vis_img.shape[:2]
        )
        
        # Draw green pentagrams for input points
        for sx, sy in scaled_coordinates:
            ax.scatter(
                sx, sy,
                marker="*",
                s=500,
                color="green",
                edgecolors="black",
                linewidth=1,
                zorder=10
            )
    
    # Save the result
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Add pentagram markers to visualization images"
    )
    parser.add_argument(
        "--vis-image",
        type=str,
        required=True,
        help="Path to the visualization image"
    )
    parser.add_argument(
        "--original-image",
        type=str,
        required=True,
        help="Path to the original image"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to CSV file with point coordinates"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the output image"
    )
    parser.add_argument(
        "--image-filename",
        type=str,
        required=True,
        help="Image filename (e.g., ISIC_0024312.jpg)"
    )
    
    args = parser.parse_args()
    
    success = add_pentagrams_to_visualization(
        args.vis_image,
        args.original_image,
        args.csv_path,
        args.output_path,
        args.image_filename
    )
    
    if success:
        print(f"✅ Successfully added pentagrams to: {args.output_path}")
    else:
        print(f"❌ Failed to add pentagrams")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())






