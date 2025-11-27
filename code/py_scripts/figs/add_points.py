import argparse
import cv2
import pandas as pd
import os


def add_circle_to_image(
    image,
    center,
    radius=20,
    color_center=(0, 255, 255),
    color_border=(0, 0, 0),
    thickness_border=5,
):
    """
    Adds a yellow circle with a black border at the specified coordinates on the image.

    :param image: Image on which to draw
    :param center: Center of the circle (x, y)
    :param radius: Radius of the circle
    :param color_center: Color of the circle center (default is yellow)
    :param color_border: Color of the circle border (default is black)
    :param thickness_border: Thickness of the border
    """
    # Draw the black border (slightly larger circle)
    cv2.circle(image, center, radius + thickness_border, color_border, thickness=-1)
    # Draw the yellow center (smaller circle)
    cv2.circle(image, center, radius, color_center, thickness=-1)


def process_image(source_file, points_path):
    """
    Reads the CSV file and processes the image to add "X" shapes.

    :param source_file: Path to the image file
    :param points_path: Path to the CSV file containing the points
    """
    # Read the CSV file
    points_data = pd.read_csv(points_path)
    print(points_path)
    # Extract the filename from the source file path
    img_filename = "_".join(os.path.basename(source_file).split("_")[2:])
    img_filename = img_filename.replace("points_", "")
    print(img_filename)
    # Find the relevant row in the CSV for this image
    row = points_data[points_data["img_filename"] == img_filename]

    if row.empty:
        print(f"Warning: No points found for image {img_filename}. Skipping.")
        return

    # Extract coordinates
    coordinates = [
        (row.iloc[0][f"y_{i}"], row.iloc[0][f"x_{i}"])
        for i in range(3)
        if pd.notnull(row.iloc[0][f"x_{i}"]) and pd.notnull(row.iloc[0][f"y_{i}"])
    ]

    # Load the image
    image = cv2.imread(source_file)
    if image is None:
        raise ValueError(f"Error loading image at {source_file}")

    # Add "X" shapes to the image
    for coord in coordinates:
        add_circle_to_image(image, coord)

    # Save the image (overwrite the original image)
    cv2.imwrite(source_file, image)
    print(f"Processed and saved image: {source_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add 'X' shapes to a single image based on coordinates from a CSV file."
    )
    parser.add_argument(
        "--source-file", type=str, required=True, help="Path to the image file."
    )
    parser.add_argument(
        "--points-path",
        type=str,
        required=True,
        help="Path to the CSV file containing the points.",
    )

    args = parser.parse_args()

    process_image(args.source_file, args.points_path)
