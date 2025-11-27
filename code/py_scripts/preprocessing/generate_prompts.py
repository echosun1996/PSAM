import json
import argparse
import os
import cv2
import numpy as np
from rich import print
from tqdm import tqdm
import random
from scipy.ndimage import binary_dilation

parser = argparse.ArgumentParser()
parser.add_argument("--img-path", type=str, default=None, help="img folder path")
parser.add_argument("--mask-path", type=str, default=None, help="mask folder path")
parser.add_argument(
    "--prompt-output", type=str, default=None, help="prompt output path"
)
parser.add_argument(
    "--vis-path", type=str, default=None, help="prompt visable output path"
)
parser.add_argument(
    "--points-sum", type=int, default=200, help="points sum for each mask"
)
parser.add_argument("--random-seed", type=int, default=42, help="random seed")
parser.add_argument(
    "--threshold",
    type=int,
    default=100,
    help="threshold for filter points (remove points close to the edge)",
)
parser.add_argument(
    "--remained-points", type=int, default=3, help="remain points number"
)


args = parser.parse_args()

random_seed = args.random_seed
img_path = args.img_path
vis_path = args.vis_path
visable = vis_path is not None

mask_path = args.mask_path
prompt_output = args.prompt_output

mask_files = os.listdir(mask_path)

points_sum = args.points_sum
threshold = args.threshold
remained_points = args.remained_points


def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def find_extreme_points(current_points):
    max_distance = -1
    min_distance = float("inf")
    farthest_point = None
    nearest_point = None
    for point in current_points:
        distance = 0
        for temp in current_points:
            if point == temp:
                continue
            distance += euclidean_distance(point, temp)
        if distance > max_distance:
            max_distance = distance
            farthest_point = point
        if distance < min_distance:
            min_distance = distance
            nearest_point = point

    return farthest_point, nearest_point


def filter_points(selected_points, mask, threshold=100):
    distance_transform = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    points_with_distances = []

    for point in selected_points:
        distance = distance_transform[point[0], point[1]]
        points_with_distances.append((point, distance))

    points_with_distances.sort(key=lambda x: x[1])  # 按照第二个元素（距离）排序

    top_distance_points = [point for point, _ in points_with_distances[-threshold:]]

    return top_distance_points


if os.path.exists(prompt_output):
    os.remove(prompt_output)


with open(prompt_output, "w") as f:
    f.write("img_filename")
    for i in range(remained_points):
        f.write(f",x_{i},y_{i}")
    f.write("\n")
    for mask_file in tqdm(mask_files, desc="Generating prompts to: " + prompt_output):
        if mask_file.endswith(".png"):
            random.seed(random_seed)
            open_path = os.path.join(mask_path, mask_file)
            mask = cv2.imread(open_path, cv2.IMREAD_GRAYSCALE)
            img_filename = mask_file.split("_segmentation.png")[0] + ".jpg"

            if visable:
                img = cv2.imread(os.path.join(img_path, img_filename), cv2.IMREAD_COLOR)
                mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            points = np.argwhere(mask == 255)

            random_indices = random.sample(
                range(len(points)), min(points_sum, len(points))
            )
            selected_points = [tuple(points[idx]) for idx in random_indices]

            selected_points = filter_points(selected_points, mask, threshold=threshold)

            if visable:
                for point in selected_points:
                    cv2.circle(mask_colored, tuple(point[::-1]), 3, (125, 125, 125), -1)

            remaining_points = selected_points[:]

            while len(remaining_points) > remained_points:
                farthest_point, nearest_point = find_extreme_points(remaining_points)

                # if farthest_point is not None and len(remaining_points) % 10 == 0:
                #     remaining_points.remove(farthest_point)
                #     if visable:
                #         cv2.putText(
                #             mask_colored,
                #             "f",
                #             tuple(farthest_point[::-1]),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             0.7,
                #             (0, 0, 255),
                #             1,
                #         )

                if nearest_point is not None and len(remaining_points) > 3:
                    remaining_points.remove(nearest_point)
                    # if visable:
                    #     cv2.putText(
                    #         mask_colored,
                    #         "n",
                    #         tuple(nearest_point[::-1]),
                    #         cv2.FONT_HERSHEY_SIMPLEX,
                    #         0.7,
                    #         (0, 0, 255),
                    #         1,
                    #     )

            final_points = remaining_points

            f.write(f"{img_filename}")
            for point in final_points:
                f.write(f",{point[0]},{point[1]}")
                # print(point)
            f.write("\n")
            f.flush()
            if visable:
                for point in final_points:
                    cv2.circle(mask_colored, tuple(point[::-1]), 10, (0, 0, 255), -1)
                    cv2.circle(img, tuple(point[::-1]), 10, (0, 0, 255), -1)
                os.makedirs(vis_path, exist_ok=True)
                cv2.imwrite(os.path.join(vis_path, mask_file), mask_colored)
                cv2.imwrite(os.path.join(vis_path, img_filename), img)
