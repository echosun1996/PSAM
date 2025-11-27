import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Set the dataset name
# class_name = "ISIC2016"
class_name = "Dermofit"

# Define paths
temp_distance_path = (
    f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))}/code/py_scripts/more_exp/temp/{class_name}"
)
figs_output_path = (
    f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))}/code/py_scripts/more_exp/output/{class_name}"
)

# Create output directory if it doesn't exist
os.makedirs(figs_output_path, exist_ok=True)


def plot_smoothness_histogram(csv_path):
    """
    Read a CSV file containing mask analysis data and plot a histogram
    of the smoothness distribution with boundaries between categories.
    平滑度值的范围均匀分为三组。

    Parameters:
    - csv_path: Path to the CSV file with analysis data
    """
    # Read the CSV file
    print(f"Reading data from {csv_path}")
    df = pd.read_csv(csv_path)

    # Check if 'smoothness' column exists
    if "smoothness" not in df.columns:
        print(f"Error: 'smoothness' column not found in {csv_path}")
        print("Available columns:", df.columns.tolist())
        return

    # Extract smoothness values
    smoothness_values = df["smoothness"].values

    # 使用鲁棒的方法确定值范围（排除极端异常值）
    # 使用1%和99%百分位数来定义有效范围，避免极端异常值的影响
    min_smoothness = np.percentile(smoothness_values, 1)  # 使用1%百分位数代替最小值
    max_smoothness = np.percentile(smoothness_values, 99)  # 使用99%百分位数代替最大值

    # 计算范围
    smoothness_range = max_smoothness - min_smoothness

    # 计算等分的阈值（将范围分成三等份）
    low_threshold = min_smoothness + smoothness_range / 3
    high_threshold = min_smoothness + 2 * smoothness_range / 3

    # Create figure
    plt.figure(figsize=(12, 8))

    # Set appealing style
    sns.set_style("whitegrid")

    # Plot histogram with KDE
    sns.histplot(
        smoothness_values,
        kde=True,
        bins=50,
        color="#3498db",
        alpha=0.7,
        edgecolor="white",
        linewidth=0.3,
    )

    # Draw vertical lines at the thresholds
    plt.axvline(
        x=low_threshold,
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        label=f"Low-Medium Threshold: {low_threshold:.2f}",
    )
    plt.axvline(
        x=high_threshold,
        color="#2ecc71",
        linestyle="--",
        linewidth=2,
        label=f"Medium-High Threshold: {high_threshold:.2f}",
    )

    # Add text annotations for the regions
    text_y_pos = plt.gca().get_ylim()[1] * 0.7

    # 计算各区域中心点的x坐标 - 均匀分布
    high_center = (min_smoothness + low_threshold) / 2
    medium_center = (low_threshold + high_threshold) / 2
    low_center = (high_threshold + max_smoothness) / 2

    plt.text(
        high_center,
        text_y_pos,
        "High Smoothness\n(More Regular)",
        ha="center",
        va="center",
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.5"),
    )

    plt.text(
        medium_center,
        text_y_pos,
        "Medium Smoothness",
        ha="center",
        va="center",
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.5"),
    )

    plt.text(
        low_center,
        text_y_pos,
        "Low Smoothness\n(More Complex)",
        ha="center",
        va="center",
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.5"),
    )

    # Calculate statistics for display
    mean_smoothness = np.mean(smoothness_values)
    median_smoothness = np.median(smoothness_values)
    std_smoothness = np.std(smoothness_values)

    # Count samples in each category
    high_count = sum(smoothness_values <= low_threshold)
    medium_count = sum(
        (smoothness_values > low_threshold) & (smoothness_values <= high_threshold)
    )
    low_count = sum(smoothness_values > high_threshold)
    total_samples = len(smoothness_values)

    # Display statistics
    stats_text = (
        f"Mean: {mean_smoothness:.2f}\n"
        f"Median: {median_smoothness:.2f}\n"
        f"Std Dev: {std_smoothness:.2f}\n"
        f"1st Percentile: {min_smoothness:.2f}\n"
        f"99th Percentile: {max_smoothness:.2f}"
    )

    group_stats = (
        f"Total samples: {total_samples}\n"
        f"High smoothness: {high_count} samples ({high_count/total_samples*100:.1f}%)\n"
        f"Medium smoothness: {medium_count} samples ({medium_count/total_samples*100:.1f}%)\n"
        f"Low smoothness: {low_count} samples ({low_count/total_samples*100:.1f}%)"
    )

    # plt.text(
    #     0.02,
    #     0.97,
    #     stats_text,
    #     transform=plt.gca().transAxes,
    #     verticalalignment="top",
    #     bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
    # )

    # plt.text(
    #     0.02,
    #     0.80,
    #     group_stats,
    #     transform=plt.gca().transAxes,
    #     verticalalignment="top",
    #     bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
    # )

    # # 在图上标注分组依据
    # plt.text(
    #     0.98,
    #     0.97,
    #     "Range-based Division\n(Equal value ranges)",
    #     transform=plt.gca().transAxes,
    #     horizontalalignment="right",
    #     verticalalignment="top",
    #     fontsize=10,
    #     bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
    # )
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)
    # Add labels and title
    plt.xlabel("Smoothness Value (Perimeter² / Area)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"Distribution of Ground Truth Smoothness", fontsize=16)

    # Add legend
    plt.legend(loc="upper right")

    # Set x-axis limits to focus on the main distribution
    plt.xlim(min_smoothness, max_smoothness)

    # Save figure
    save_path = os.path.join(
        figs_output_path, f"{class_name}_smoothness_histogram_equal_range.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Histogram saved to {save_path}")

    # Display category counts
    print("\nCategory Counts (Equal range division):")
    print(
        f"High Smoothness (≤{low_threshold:.2f}): {high_count} masks ({high_count/total_samples*100:.1f}%)"
    )
    print(
        f"Medium Smoothness ({low_threshold:.2f}-{high_threshold:.2f}): {medium_count} masks ({medium_count/total_samples*100:.1f}%)"
    )
    print(
        f"Low Smoothness (>{high_threshold:.2f}): {low_count} masks ({low_count/total_samples*100:.1f}%)"
    )
    print(
        f"\nSmothness range: {min_smoothness:.2f} to {max_smoothness:.2f} (excluding extreme outliers)"
    )
    print(f"Each group spans approximately {smoothness_range/3:.2f} units")

    # Show plot
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the analysis."""
    # Define the CSV file path
    csv_file = os.path.join(temp_distance_path, f"{class_name}_analysis_data.csv")

    # Check if the file exists
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found at {csv_file}")
        # Try to look for any CSV files in the directory
        csv_files = list(Path(temp_distance_path).glob("*.csv"))
        if csv_files:
            print("Found these CSV files instead:")
            for file in csv_files:
                print(f"- {file.name}")
            csv_file = str(csv_files[0])  # Use the first one
            print(f"Using {csv_file} instead.")
        else:
            print("No CSV files found in the directory.")
            return

    # Plot the smoothness histogram
    plot_smoothness_histogram(csv_file)


if __name__ == "__main__":
    main()
