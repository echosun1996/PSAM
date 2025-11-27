import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
import glob
from pathlib import Path
import random
from tqdm import tqdm
import cv2
from sklearn.cluster import KMeans
from scipy import stats

# Calculate the distribution of the Hausdorff Distance Distribution by Smoothness Category.
# class_name = "ISIC2016"
# class_name = "Dermofit"
# class_name = "HAM10000"
class_name = "AtlasZoomIn10"


# Define paths based on class_name
# Use relative path or environment variable
parent_dir = os.environ.get('PSAM_PARENT_DIR', os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
output_mask_path = f"{parent_dir}/results/PSAM/{class_name}/test/mask"

baseline_mask_path = (
    f"{parent_dir}/data/{class_name}/output/test/OnePrompt_points"
)
ground_truth_path = (
    f"{parent_dir}/data/{class_name}/input/test/{class_name}_seg"
)
temp_distance_path = (
    f"{parent_dir}/code/py_scripts/more_exp/temp/{class_name}"
)
figs_output_path = (
    f"{parent_dir}/code/py_scripts/more_exp/output/{class_name}"
)

# Create directories if they don't exist
os.makedirs(temp_distance_path, exist_ok=True)
os.makedirs(figs_output_path, exist_ok=True)


def calculate_smoothness(mask_path):
    """
    Calculate the smoothness of a mask region using the ratio of perimeter squared to area.
    Lower values indicate smoother shapes (circle has the minimum value).
    """
    # Read the image
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Resize to 64x64 to speed up computation
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)

    # Ensure binary mask
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Take the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate perimeter and area
    perimeter = cv2.arcLength(largest_contour, True)
    area = cv2.contourArea(largest_contour)

    if area == 0:
        return None

    # Calculate smoothness as perimeter^2 / area
    # For a perfect circle, this value is 4π (minimum possible value)
    # Higher values indicate less smooth (more complex) shapes
    smoothness = (perimeter**2) / area

    return smoothness


def load_and_preprocess_mask(mask_path):
    """Load and preprocess mask to binary numpy array, resizing to 64x64."""
    mask = Image.open(mask_path).convert("L")
    # Resize to 64x64 to speed up computation
    mask = mask.resize((64, 64), Image.NEAREST)
    mask_array = np.array(mask) > 0  # Convert to binary
    return mask_array.astype(np.uint8) * 255  # Return as numpy array64."""


def dice_coefficient(mask1, mask2):
    # Ensure the masks are binary (0 or 1)
    mask1 = np.asarray(mask1).astype(np.bool8)
    mask2 = np.asarray(mask2).astype(np.bool8)

    # Calculate intersection and union
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1) + np.sum(mask2)

    # Return the Dice coefficient
    return (
        2.0 * intersection / union if union != 0 else 1.0
    )  # To handle cases where both are empty


# Relationship between Mask Smoothness and Hausdorff Distance
def analyze_masks():
    """Analyze the smoothness of ground truth masks and calculate Hausdorff distances."""
    # Get ground truth file list
    gt_files = sorted(glob.glob(os.path.join(ground_truth_path, "*_segmentation.png")))

    analysis_data = []

    # Use tqdm for progress tracking
    for gt_file in tqdm(gt_files, desc="Analyzing masks"):
        # Extract image ID
        filename = os.path.basename(gt_file)
        image_id = filename.split("_segmentation")[0]  # ISIC_0000003

        # Find corresponding prediction file
        pred_file = os.path.join(output_mask_path, f"{image_id}.jpg")
        baseline_file = os.path.join(baseline_mask_path, f"{image_id}.jpg")

        if not os.path.exists(pred_file):
            print(f"Prediction not found for {image_id}, skipping...")
            continue

        try:
            # Calculate smoothness for ground truth mask
            smoothness = calculate_smoothness(gt_file)

            if smoothness is None:
                print(f"Could not calculate smoothness for {image_id}, skipping...")
                continue

            # Calculate Hausdorff distance
            baseline_mask = load_and_preprocess_mask(baseline_file)
            gt_mask = load_and_preprocess_mask(gt_file)
            pred_mask = load_and_preprocess_mask(pred_file)

            # Calculate Dice coefficient for ground truth and predicted mask
            dice_score_predict = dice_coefficient(gt_mask, pred_mask)
            dice_score_baseline = dice_coefficient(gt_mask, baseline_mask)

            # Store results
            analysis_data.append(
                {
                    "image_id": image_id,
                    "smoothness": smoothness,
                    "dice_score_predict": dice_score_predict,
                    "dice_score_baseline": dice_score_baseline,
                }
            )

        except Exception as e:
            print(f"Error processing {image_id}: {str(e)}")

    # Save analysis data to CSV
    csv_path = os.path.join(temp_distance_path, f"{class_name}_analysis_data.csv")
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "image_id",
            "smoothness",
            "dice_score_predict",
            "dice_score_baseline",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in analysis_data:
            writer.writerow(row)

    print(f"Analysis data saved to {csv_path}")
    return analysis_data


def categorize_by_smoothness(analysis_data):
    """
    Categorize masks into low, medium, and high smoothness based on equal value ranges.
    将平滑度值的范围均匀分为三组，而不是按照样本数量分组。

    Parameters:
    - analysis_data: List of dictionaries containing analysis results

    Returns:
    - high_smoothness: List of items with high smoothness (more regular shapes)
    - medium_smoothness: List of items with medium smoothness
    - low_smoothness: List of items with low smoothness (more complex shapes)
    """
    if not analysis_data:
        return [], [], []

    # Extract smoothness values
    smoothness_values = [d["smoothness"] for d in analysis_data]

    # 使用鲁棒的方法确定值范围（排除极端异常值）
    # 使用1%和99%百分位数来定义有效范围，避免极端异常值的影响
    min_smoothness = np.percentile(smoothness_values, 1)  # 使用1%百分位数代替最小值
    max_smoothness = np.percentile(smoothness_values, 99)  # 使用99%百分位数代替最大值

    # 计算范围
    smoothness_range = max_smoothness - min_smoothness

    # 计算等分的阈值（将范围分成三等份）
    low_threshold = min_smoothness + smoothness_range / 3
    high_threshold = min_smoothness + 2 * smoothness_range / 3

    # Lower smoothness values indicate smoother shapes
    high_smoothness = [d for d in analysis_data if d["smoothness"] <= low_threshold]
    medium_smoothness = [
        d for d in analysis_data if low_threshold < d["smoothness"] <= high_threshold
    ]
    low_smoothness = [d for d in analysis_data if d["smoothness"] > high_threshold]

    # Count samples in each category
    high_count = len(high_smoothness)
    medium_count = len(medium_smoothness)
    low_count = len(low_smoothness)
    total_samples = len(analysis_data)

    print(
        f"Categorized by equal value ranges:"
        f"\nHigh smoothness (≤{low_threshold:.2f}): {high_count} samples ({high_count/total_samples*100:.1f}%)"
        f"\nMedium smoothness ({low_threshold:.2f}-{high_threshold:.2f}): {medium_count} samples ({medium_count/total_samples*100:.1f}%)"
        f"\nLow smoothness (>{high_threshold:.2f}): {low_count} samples ({low_count/total_samples*100:.1f}%)"
        f"\nSmothness range: {min_smoothness:.2f} to {max_smoothness:.2f} (excluding extreme outliers)"
        f"\nEach group spans approximately {smoothness_range/3:.2f} units"
    )

    return high_smoothness, medium_smoothness, low_smoothness


def plot_combined_boxplot(categories, category_names):
    """
    Create combined boxplot comparing our method vs baseline across all smoothness categories.
    Uses standard rectangular boxes, shows mean values inside boxes, and adds percentage improvement labels.
    X-axis labels are displayed horizontally with category names at the bottom.

    Parameters:
    - categories: List of lists containing data for each smoothness category
    - category_names: List of names for each category
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare boxplot data
    all_data = []
    original_data = []  # Save original data for connecting lines
    box_labels = []  # Individual box labels

    # For each category (high, medium, low smoothness)
    for i, category in enumerate(categories):
        if category:  # Check if category is not empty
            # Extract data for our method
            predict_data = [item["dice_score_predict"] for item in category]
            all_data.append(predict_data)
            # Extract data for baseline method
            baseline_data = [item["dice_score_baseline"] for item in category]
            all_data.append(baseline_data)

            # Save original data
            original_data.append(
                {
                    "our_method": predict_data,
                    "baseline": baseline_data,
                    "items": category,
                }
            )

            # Add individual box labels
            box_labels.append("PSAM")
            box_labels.append("Baseline")

    # Create boxplot, with standard rectangular boxes (notch=False) and no outliers
    bp = ax.boxplot(all_data, patch_artist=True, notch=False, showfliers=False)

    # Customize boxplot colors
    colors = []
    for i in range(len(categories)):
        colors.extend(["#3498db", "#e74c3c"])  # Blue for our method, red for baseline

    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=colors[i], alpha=0.7)

    # Change median lines to black
    for median in bp["medians"]:
        median.set(color="black", linewidth=1.5)

    # Customize plot
    ax.set_title(
        f"Dice Coefficient Comparison between PSAM and Baseline",
        fontsize=16,
    )
    ax.set_ylabel("Dice Coefficient", fontsize=12)

    # Set individual box labels horizontally
    ax.set_xticklabels(box_labels, fontsize=10)

    # Add category names at the bottom
    category_positions = []
    for i in range(len(categories)):
        if categories[i]:  # Only for non-empty categories
            # Calculate the middle position for each category
            middle_pos = i * 2 + 1.5
            category_positions.append(middle_pos)

    # Create a second x-axis for category names
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(category_positions)
    ax2.set_xticklabels(
        [
            f"{name} Smoothness"
            for name in category_names
            if categories[category_names.index(name)]
        ],
        fontsize=12,
        fontweight="bold",
    )
    # Move the second axis to the bottom
    ax2.spines["bottom"].set_position(("outward", 40))
    ax2.spines["top"].set_visible(False)
    ax2.xaxis.set_ticks_position("bottom")

    # Add gaps between category label lines
    for i, pos in enumerate(category_positions):
        if i > 0:  # Skip first tick
            # Remove default tick line
            ax2.xaxis.get_major_ticks()[i].tick1line.set_visible(False)
            ax2.xaxis.get_major_ticks()[i].tick2line.set_visible(False)

            # Draw custom tick line with gap
            line_x = pos
            line_y_start = (
                ax2.get_transform().transform((line_x, 0))[1] - 5
            )  # Bottom of tick
            line_y_end = line_y_start - 8  # Top of tick

            # Draw tick line with gap in middle
            gap_size = 4  # Size of the gap
            mid_point = (line_y_start + line_y_end) / 2

            # Draw top part
            ax.annotate(
                "",
                xy=(line_x, mid_point - gap_size / 2),
                xytext=(line_x, line_y_end),
                xycoords="data",
                arrowprops=dict(arrowstyle="-", color="black", linewidth=1),
            )

            # Draw bottom part
            ax.annotate(
                "",
                xy=(line_x, mid_point + gap_size / 2),
                xytext=(line_x, line_y_start),
                xycoords="data",
                arrowprops=dict(arrowstyle="-", color="black", linewidth=1),
            )

    ax.grid(True, linestyle="--", alpha=0.7)

    # Add mean values inside boxes
    for i, data in enumerate(all_data):
        mean_val = np.mean(data)
        # Get current box y-coordinate midpoint
        box = bp["boxes"][i]
        box_y_mean = np.mean([p[1] for p in box.get_path().vertices])
        # Add mean label inside box - without black border
        ax.text(
            i + 1,
            box_y_mean,
            f"Mean\n{mean_val:.3f}",
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(
                boxstyle="round", facecolor="white", alpha=0.7, edgecolor="white"
            ),
        )

    # Add percentage improvement labels next to "Our Method" boxes
    for i in range(len(categories)):
        if i < len(original_data):  # Safety check
            our_method_idx = i * 2
            our_data = all_data[our_method_idx]
            baseline_data = all_data[our_method_idx + 1]

            our_mean = np.mean(our_data)
            baseline_mean = np.mean(baseline_data)

            # Calculate percentage improvement
            pct_improvement = ((our_mean - baseline_mean) / baseline_mean) * 100

            # Position the text next to our method box
            box = bp["boxes"][our_method_idx]
            box_x = our_method_idx + 1  # Box position in the plot
            box_y = np.max([p[1] for p in box.get_path().vertices])  # Top of the box

            # Add the percentage text with appropriate color and sign
            if pct_improvement > 0:
                color = "green"
                sign = "+"
            else:
                color = "red"
                sign = ""  # Negative sign will be included in the number

            ax.text(
                box_x + 0.25,
                box_y,
                f"{sign}{pct_improvement:.2f}%",
                color=color,
                fontweight="bold",
                ha="left",
                va="center",
                fontsize=9,
            )

    # Create legend
    import matplotlib.patches as mpatches

    our_patch = mpatches.Patch(color="#3498db", alpha=0.7, label="PSAM")
    baseline_patch = mpatches.Patch(color="#e74c3c", alpha=0.7, label="Baseline")
    ax.legend(handles=[our_patch, baseline_patch], loc="lower right")

    # Add dividing lines between groups
    for i in range(1, len(categories)):
        if categories[i]:  # Only draw line if category is not empty
            line_position = i * 2 + 0.5
            ax.axvline(x=line_position, color="black", linestyle="--", alpha=0.5)

    # Draw light dotted lines between paired samples
    for i, category_data in enumerate(original_data):
        our_method_x = i * 2 + 1
        baseline_x = i * 2 + 2
        our_data = category_data["our_method"]
        baseline_data = category_data["baseline"]

        # Calculate point positions with small jitter
        np.random.seed(42)  # For reproducibility
        our_x = np.random.normal(our_method_x, 0.05, size=len(our_data))
        baseline_x = np.random.normal(baseline_x, 0.05, size=len(baseline_data))

        # Add semi-transparent points
        our_scatter = ax.scatter(
            our_x, our_data, s=10, alpha=0.4, color="#3498db", zorder=1
        )
        baseline_scatter = ax.scatter(
            baseline_x, baseline_data, s=10, alpha=0.4, color="#e74c3c", zorder=1
        )

        # Connect corresponding points
        for j in range(len(our_data)):
            ax.plot(
                [our_x[j], baseline_x[j]],
                [our_data[j], baseline_data[j]],
                color="gray",
                linestyle=":",
                alpha=0.2,
                linewidth=0.5,
                zorder=0,
            )

    # Adjust layout to accommodate the second x-axis
    plt.subplots_adjust(bottom=0.2)

    # Save figure
    save_path = os.path.join(
        figs_output_path, f"{class_name}_dice_standard_boxplot_horizontal_labels.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Standard boxplot with horizontal labels saved to {save_path}")

    # Show figure
    plt.show()


def analyze_and_report_statistics(categories, category_names):
    """
    Analyze and report statistical differences between methods for each category.

    Parameters:
    - categories: List of lists containing data for each smoothness category
    - category_names: List of names for each category
    """
    print("\n" + "=" * 60)
    print(f"Statistical Analysis for {class_name}")
    print("=" * 60)

    # Analyze each category
    for i, category in enumerate(categories):
        if not category:
            continue

        # Extract scores
        predict_scores = [item["dice_score_predict"] for item in category]
        baseline_scores = [item["dice_score_baseline"] for item in category]

        # Calculate basic statistics
        predict_mean = np.mean(predict_scores)
        baseline_mean = np.mean(baseline_scores)
        predict_median = np.median(predict_scores)
        baseline_median = np.median(baseline_scores)

        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(predict_scores, baseline_scores)

        # Report findings
        print(f"\nCategory: {category_names[i]} Smoothness (n={len(category)})")
        print(f"PSAM: Mean={predict_mean:.4f}, Median={predict_median:.4f}")
        print(f"Baseline:   Mean={baseline_mean:.4f}, Median={baseline_median:.4f}")
        print(
            f"Improvement: {(predict_mean - baseline_mean) / baseline_mean * 100:.2f}%"
        )
        print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.6f}")

        # Interpret the result
        if p_value < 0.05:
            if predict_mean > baseline_mean:
                print(
                    "Result: PSAM performs SIGNIFICANTLY BETTER than the baseline (p<0.05)"
                )
            else:
                print(
                    "Result: Baseline performs SIGNIFICANTLY BETTER than our method (p<0.05)"
                )
        else:
            print(
                "Result: No statistically significant difference between methods (p≥0.05)"
            )

    print("\n" + "=" * 60)


def main():
    print(f"Starting analysis for {class_name}...")

    # Analyze masks and calculate Hausdorff distances
    analysis_data = analyze_masks()

    # Categorize masks based on ground truth smoothness - using equal value ranges
    high_smoothness, medium_smoothness, low_smoothness = categorize_by_smoothness(
        analysis_data
    )
    categories = [high_smoothness, medium_smoothness, low_smoothness]
    category_names = ["High", "Medium", "Low"]

    print(
        f"Found {len(high_smoothness)} high smoothness, {len(medium_smoothness)} medium smoothness, and {len(low_smoothness)} low smoothness masks."
    )

    # Create only the improved boxplot with connections between points
    plot_combined_boxplot(categories, category_names)

    # Analyze and report statistical differences
    analyze_and_report_statistics(categories, category_names)

    print(f"Analysis complete for {class_name}!")


if __name__ == "__main__":
    main()
