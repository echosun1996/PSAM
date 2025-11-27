"""
Script to extract positions of three-color pentagram markers from token visualization images.

The three colors represent:
- Green: Input points (from CSV file)
- Pink/Magenta (#dc307e): Positive supplement points (from _ps.png mask files)
- Blue (#1c1cf5): Negative supplement points (from _ns.png mask files)
"""

import os
import cv2
import csv
import numpy as np
import argparse
from pathlib import Path


def extract_input_points_from_csv(csv_path, image_filename):
    """
    Extract input point coordinates from CSV file.
    
    Args:
        csv_path: Path to the CSV file containing point coordinates
        image_filename: Name of the image file (e.g., "ISIC_0012086.jpg")
    
    Returns:
        List of [x, y] coordinates for input points
    """
    coordinates = []
    with open(csv_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["img_filename"] == image_filename:
                # Read coordinates (assuming 3 points based on uncertain_vis.py)
                for i in range(3):
                    try:
                        x = float(row[f"x_{i}"])
                        y = float(row[f"y_{i}"])
                        coordinates.append([x, y])
                    except KeyError:
                        break
                return coordinates
    return coordinates


def extract_points_from_mask(mask_path, original_image_shape, mask_shape):
    """
    Extract point coordinates from a binary mask image.
    
    Args:
        mask_path: Path to the mask image (_ps.png or _ns.png)
        original_image_shape: Shape of the original image (height, width)
        mask_shape: Shape of the mask image (height, width)
    
    Returns:
        List of [x, y] coordinates scaled to original image size
    """
    if not os.path.exists(mask_path):
        return []
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    
    # Find all non-zero pixels in the mask
    points = cv2.findNonZero(mask)
    if points is None:
        return []
    
    coordinates = []
    for point in points:
        x, y = point[0]
        # Scale coordinates to match original image size
        adjusted_x = int(x * original_image_shape[1] / mask.shape[1])
        adjusted_y = int(y * original_image_shape[0] / mask.shape[0])
        coordinates.append([adjusted_x, adjusted_y])
    
    return coordinates


def extract_all_pentagram_positions(mask_dir, csv_path, image_filename, image_shape=(1024, 1024)):
    """
    Extract all three types of pentagram marker positions.
    
    Args:
        mask_dir: Directory containing mask files (_ps.png, _ns.png, etc.)
        csv_path: Path to CSV file with input point coordinates
        image_filename: Name of the image file (e.g., "ISIC_0012086.jpg")
        image_shape: Shape of the original image (height, width)
    
    Returns:
        Dictionary with keys: 'input_points' (green), 'positive_points' (pink), 
        'negative_points' (blue)
    """
    base_name = os.path.splitext(image_filename)[0]
    
    # Extract input points from CSV (green pentagrams)
    input_points = extract_input_points_from_csv(csv_path, image_filename)
    
    # Extract positive supplement points from _ps.png (pink pentagrams)
    ps_mask_path = os.path.join(mask_dir, f"{base_name}_ps.png")
    positive_points = extract_points_from_mask(ps_mask_path, image_shape, image_shape)
    
    # Extract negative supplement points from _ns.png (blue pentagrams)
    ns_mask_path = os.path.join(mask_dir, f"{base_name}_ns.png")
    negative_points = extract_points_from_mask(ns_mask_path, image_shape, image_shape)
    
    return {
        'input_points': input_points,  # Green pentagrams
        'positive_points': positive_points,  # Pink/Magenta pentagrams (#dc307e)
        'negative_points': negative_points  # Blue pentagrams (#1c1cf5)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract positions of three-color pentagram markers from token visualization"
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        required=True,
        help="Directory containing mask files (_ps.png, _ns.png, etc.)"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to CSV file containing input point coordinates"
    )
    parser.add_argument(
        "--image-filename",
        type=str,
        required=True,
        help="Name of the image file (e.g., ISIC_0012086.jpg)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path to save positions (JSON format). If not specified, prints to stdout"
    )
    parser.add_argument(
        "--image-shape",
        type=int,
        nargs=2,
        default=[1024, 1024],
        help="Shape of the original image [height width] (default: 1024 1024)"
    )
    
    args = parser.parse_args()
    
    # Extract all positions
    positions = extract_all_pentagram_positions(
        args.mask_dir,
        args.csv_path,
        args.image_filename,
        tuple(args.image_shape)
    )
    
    # Format output
    output_data = {
        'image_filename': args.image_filename,
        'image_shape': args.image_shape,
        'positions': {
            'input_points': {
                'color': 'green',
                'count': len(positions['input_points']),
                'coordinates': positions['input_points']
            },
            'positive_supplement_points': {
                'color': '#dc307e',
                'count': len(positions['positive_points']),
                'coordinates': positions['positive_points']
            },
            'negative_supplement_points': {
                'color': '#1c1cf5',
                'count': len(positions['negative_points']),
                'coordinates': positions['negative_points']
            }
        }
    }
    
    # Save or print results
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Positions saved to: {args.output}")
    else:
        import json
        print(json.dumps(output_data, indent=2))
    
    # Print summary
    print("\nSummary:")
    print(f"  Green (Input) points: {len(positions['input_points'])}")
    print(f"  Pink (Positive supplement) points: {len(positions['positive_points'])}")
    print(f"  Blue (Negative supplement) points: {len(positions['negative_points'])}")


if __name__ == "__main__":
    main()






