import argparse
import os
import cv2
import numpy as np


def overlay_masks(source_path, pred_mask_path, gt_mask_path, save_dir):
    # Load the images
    source_img = cv2.imread(source_path)
    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
    # 判断图像是否全是黑色
    if np.all(pred_mask == 0):
        # 如果全是黑色，则将图像全部变为白色
        pred_mask[:] = 255  # 将所有像素值设置为255

    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

    # Check if the images are loaded properly
    if source_img is None or pred_mask is None or gt_mask is None:
        print("Error: One or more images could not be loaded.")
        return

    # Find contours of the predicted mask and ground truth mask
    pred_contours, _ = cv2.findContours(
        pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    gt_contours, _ = cv2.findContours(
        gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw the predicted contours in red on the source image
    cv2.drawContours(source_img, pred_contours, -1, (255, 255, 255), 14)
    # White base for glow
    cv2.drawContours(source_img, pred_contours, -1, (0, 0, 255), 10)  # Red color in BGR
    # cv2.drawContours(source_img, pred_contours, -1, (8, 7, 244), 10)  # #f40708

    # Draw the ground truth contours in green on the source image
    cv2.drawContours(source_img, gt_contours, -1, (255, 255, 255), 11)
    cv2.drawContours(source_img, gt_contours, -1, (0, 255, 0), 7)  # Green color in BGR
    # cv2.drawContours(source_img, gt_contours, -1, (238, 239, 97), 7)  # #61efee

    # Construct the save path
    save_path = os.path.join(save_dir, os.path.basename(pred_mask_path))

    # Save the resulting image
    cv2.imwrite(save_path, source_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlay predicted and ground truth masks on the source image."
    )
    parser.add_argument(
        "--source-path", type=str, required=True, help="Path to the source image."
    )
    parser.add_argument(
        "--pred-mask-path",
        type=str,
        required=True,
        help="Path to the predicted mask image.",
    )
    parser.add_argument(
        "--gt-mask-path",
        type=str,
        required=True,
        help="Path to the ground truth mask image.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory to save the resulting image.",
    )

    args = parser.parse_args()

    overlay_masks(
        args.source_path, args.pred_mask_path, args.gt_mask_path, args.save_dir
    )
