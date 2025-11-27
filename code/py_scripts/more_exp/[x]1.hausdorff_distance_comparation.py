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


# Define class name
# class_name = "ISIC2016"
# class_name = "HAM10000"
class_name = "Dermofit"

# Define paths based on class_name
import os
parent_dir = os.environ.get('PSAM_PARENT_DIR', os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
output_mask_path = f"{parent_dir}/results/PSAM/{class_name}/test/mask"
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


def perform_pca_visualization(category, category_name):
    """Perform PCA visualization for a category."""
    from tqdm import tqdm
    from sklearn.cluster import KMeans  # Make sure to import KMeans

    if not category:
        print(f"No images in {category_name} category, skipping PCA...")
        return

    # Sample up to 100 files proportionally
    sample_size = min(100, len(category))

    # Use random sampling without replacement
    sampled_items = random.sample(category, sample_size)

    print(
        f"Sampling {sample_size} files from {len(category)} total files in {category_name} category"
    )

    gt_features = []
    pred_features = []
    image_ids = []
    hausdorff_distances = []
    smoothness_values = []

    # Use tqdm for progress tracking
    for item in tqdm(sampled_items, desc=f"Processing {category_name} category"):
        image_id = item["image_id"]
        image_ids.append(image_id)
        hausdorff_distances.append(item["hausdorff_distance"])
        smoothness_values.append(item["smoothness"])

        gt_file = os.path.join(ground_truth_path, f"{image_id}_segmentation.png")
        pred_file = os.path.join(output_mask_path, f"{image_id}.jpg")

        if os.path.exists(gt_file) and os.path.exists(pred_file):
            gt_features.append(extract_features(gt_file))
            pred_features.append(extract_features(pred_file))

    if not gt_features or not pred_features:
        print(f"No valid features extracted for {category_name} category")
        return

    # Make sure all feature vectors have the same length by padding or truncating
    max_length = max(len(f) for f in gt_features + pred_features)

    # Pad or truncate features
    gt_features_padded = [
        (
            np.pad(f, (0, max_length - len(f)), "constant")
            if len(f) < max_length
            else f[:max_length]
        )
        for f in gt_features
    ]
    pred_features_padded = [
        (
            np.pad(f, (0, max_length - len(f)), "constant")
            if len(f) < max_length
            else f[:max_length]
        )
        for f in pred_features
    ]

    # Stack features
    all_features = np.vstack(gt_features_padded + pred_features_padded)

    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_features)

    # Split results back into ground truth and predictions
    gt_pca = pca_result[: len(gt_features)]
    pred_pca = pca_result[len(gt_features) :]

    # ----- 1. Standard PCA visualization (original style) -----
    plt.figure(figsize=(12, 9))

    # Ground truth points
    plt.scatter(
        gt_pca[:, 0],
        gt_pca[:, 1],
        label="Ground Truth",
        alpha=0.7,
        marker="o",
        color="blue",
    )

    # Prediction points
    plt.scatter(
        pred_pca[:, 0],
        pred_pca[:, 1],
        label="Prediction",
        alpha=0.7,
        marker="x",
        color="red",
    )

    # Draw lines connecting corresponding points
    for i in range(len(gt_pca)):
        plt.plot(
            [gt_pca[i, 0], pred_pca[i, 0]],
            [gt_pca[i, 1], pred_pca[i, 1]],
            "k-",
            alpha=0.3,
        )

    plt.title(f"PCA Visualization for {category_name} Smoothness Category", fontsize=20)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2f}%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add variance explained text in top right corner
    total_var = sum(pca.explained_variance_ratio_)
    plt.text(
        0.95,
        0.95,
        f"Total variance explained: {total_var:.2f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    # Save figure
    fig_path = os.path.join(
        figs_output_path, f"{class_name}_pca_{category_name.lower()}_smoothness.png"
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved standard PCA visualization to {fig_path}")

    # ----- 2. PCA with Hausdorff distance color mapping -----
    plt.figure(figsize=(10, 8))

    # Ground truth points colored by Hausdorff distance
    norm = plt.Normalize(min(hausdorff_distances), max(hausdorff_distances))

    # Ground truth points
    gt_scatter = plt.scatter(
        gt_pca[:, 0],
        gt_pca[:, 1],
        c=hausdorff_distances,
        cmap="plasma",
        alpha=0.7,
        marker="o",
        norm=norm,
    )

    # Prediction points - use the same colors
    pred_scatter = plt.scatter(
        pred_pca[:, 0],
        pred_pca[:, 1],
        c=hausdorff_distances,
        cmap="plasma",
        alpha=0.7,
        marker="x",
        norm=norm,
    )

    # Draw lines connecting corresponding points
    for i in range(len(gt_pca)):
        plt.plot(
            [gt_pca[i, 0], pred_pca[i, 0]],
            [gt_pca[i, 1], pred_pca[i, 1]],
            "k-",
            alpha=0.3,
        )

    plt.title(
        f"PCA Visualization for {category_name} Smoothness\nColored by Hausdorff Distance"
    )
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2f}%)")
    plt.colorbar(gt_scatter, label="Hausdorff Distance")
    plt.legend(["Ground Truth (o)", "Prediction (x)"])
    plt.grid(True, alpha=0.3)

    # Add variance explained text in top right corner
    plt.text(
        0.95,
        0.95,
        f"Total variance explained: {total_var:.2f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    # Save figure
    hd_fig_path = os.path.join(
        figs_output_path,
        f"{class_name}_pca_{category_name.lower()}_smoothness_hd_color.png",
    )
    plt.savefig(hd_fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved Hausdorff-colored PCA visualization to {hd_fig_path}")

    # Perform clustering on the hausdorff distances
    # This is what was missing - need to call the cluster_hausdorff_distances function
    category_with_clusters, cluster_centers = cluster_hausdorff_distances(
        sampled_items, n_clusters=3
    )

    # Extract clusters from the modified data
    clusters = [item["cluster"] for item in category_with_clusters]

    # Create additional visualizations for detailed analysis
    # Show examples from each category
    if len(image_ids) > 0:
        num_examples = min(3, len(image_ids))
        fig, axes = plt.subplots(num_examples, 2, figsize=(10, 5 * num_examples))

        if num_examples == 1:
            axes = np.array([axes])

        # Sort by Hausdorff distance
        sorted_indices = np.argsort(hausdorff_distances)

        for i in range(num_examples):
            idx = sorted_indices[
                i * len(sorted_indices) // num_examples
            ]  # Evenly distribute examples
            image_id = image_ids[idx]
            hd = hausdorff_distances[idx]

            gt_file = os.path.join(ground_truth_path, f"{image_id}_segmentation.png")
            pred_file = os.path.join(output_mask_path, f"{image_id}.jpg")

            axes[i, 0].imshow(np.array(Image.open(gt_file)), cmap="gray")
            axes[i, 0].set_title(f"Ground Truth: {image_id}\nHD: {hd:.2f}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(np.array(Image.open(pred_file)), cmap="gray")
            axes[i, 1].set_title(f"Prediction: {image_id}")
            axes[i, 1].axis("off")

        plt.suptitle(f"{category_name} Smoothness Category Examples")
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Make room for suptitle

        examples_fig_path = os.path.join(
            figs_output_path,
            f"{class_name}_examples_{category_name.lower()}_smoothness.png",
        )
        plt.savefig(examples_fig_path, dpi=300, bbox_inches="tight")
        print(f"Saved example visualizations to {examples_fig_path}")

    # Create additional visualization showing examples from each cluster
    for cluster_id in set(clusters):
        cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]

        if not cluster_indices:
            continue

        # Display the top 3 examples from each cluster (or fewer if less than 3)
        num_examples = min(3, len(cluster_indices))

        fig, axes = plt.subplots(num_examples, 2, figsize=(10, 5 * num_examples))
        if num_examples == 1:
            axes = np.array([axes])

        # Sort by Hausdorff distance within cluster
        cluster_items = [(i, hausdorff_distances[i]) for i in cluster_indices]
        sorted_items = sorted(cluster_items, key=lambda x: x[1])

        for i in range(num_examples):
            if i < len(sorted_items):
                idx, hd = sorted_items[i]
                image_id = image_ids[idx]

                gt_file = os.path.join(
                    ground_truth_path, f"{image_id}_segmentation.png"
                )
                pred_file = os.path.join(output_mask_path, f"{image_id}.jpg")

                axes[i, 0].imshow(np.array(Image.open(gt_file)), cmap="gray")
                axes[i, 0].axis("off")

                axes[i, 1].imshow(np.array(Image.open(pred_file)), cmap="gray")
                axes[i, 1].axis("off")

                if i == 0:
                    axes[i, 0].set_title(f"Ground Truth")
                    axes[i, 1].set_title(f"Prediction")

        plt.suptitle(
            # f"{category_name} Smoothness - Cluster {cluster_id} Examples (HD ≈ {cluster_centers[cluster_id]:.2f})"
            f"{category_name} Smoothness",
            fontsize=20,
        )
        plt.tight_layout()
        # plt.subplots_adjust(top=0.92)  # Make room for suptitle

        # Save figure
        cluster_fig_path = os.path.join(
            figs_output_path,
            f"{class_name}_{category_name.lower()}_smoothness_cluster{cluster_id}_examples.png",
        )
        plt.savefig(cluster_fig_path, dpi=300, bbox_inches="tight")
        print(f"Saved cluster {cluster_id} examples to {cluster_fig_path}")


def load_and_preprocess_mask(mask_path):
    """Load and preprocess mask to binary numpy array, resizing to 64x64."""
    mask = Image.open(mask_path).convert("L")
    # Resize to 64x64 to speed up computation
    mask = mask.resize((64, 64), Image.NEAREST)
    mask_array = np.array(mask) > 0  # Convert to binary
    return mask_array.astype(np.uint8) * 255  # Return as numpy array64."""
    mask = Image.open(mask_path).convert("L")
    # Resize to 64x64 to speed up computation
    mask = mask.resize((64, 64), Image.NEAREST)
    mask_array = np.array(mask) > 0  # Convert to binary
    return (
        torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).float()
    )  # Shape: [1, 1, 64, 64]


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
            gt_mask = load_and_preprocess_mask(gt_file)
            pred_mask = load_and_preprocess_mask(pred_file)

            # Calculate Hausdorff distance using OpenCV
            gt_contours, _ = cv2.findContours(
                gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            pred_contours, _ = cv2.findContours(
                pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not gt_contours or not pred_contours:
                print(f"No contours found for {image_id}, skipping...")
                continue

            # Take the largest contours
            gt_contour = max(gt_contours, key=cv2.contourArea)
            pred_contour = max(pred_contours, key=cv2.contourArea)

            # Calculate Hausdorff distance
            hd = cv2.matchShapes(gt_contour, pred_contour, cv2.CONTOURS_MATCH_I3, 0)

            # Store results
            analysis_data.append(
                {
                    "image_id": image_id,
                    "smoothness": smoothness,
                    "hausdorff_distance": hd,
                }
            )

        except Exception as e:
            print(f"Error processing {image_id}: {str(e)}")

    # Save analysis data to CSV
    csv_path = os.path.join(temp_distance_path, f"{class_name}_analysis_data.csv")
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["image_id", "smoothness", "hausdorff_distance"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in analysis_data:
            writer.writerow(row)

    print(f"Analysis data saved to {csv_path}")
    return analysis_data


def categorize_by_smoothness(analysis_data):
    """Categorize masks into low, medium, and high smoothness."""
    if not analysis_data:
        return [], [], []

    # Extract smoothness values
    smoothness_values = [d["smoothness"] for d in analysis_data]

    # Calculate thresholds using percentiles
    # We use 33rd and 66th percentiles for even distribution
    low_threshold = np.percentile(smoothness_values, 33)
    high_threshold = np.percentile(smoothness_values, 66)

    # Lower smoothness values indicate smoother shapes
    high_smoothness = [d for d in analysis_data if d["smoothness"] <= low_threshold]
    medium_smoothness = [
        d for d in analysis_data if low_threshold < d["smoothness"] <= high_threshold
    ]
    low_smoothness = [d for d in analysis_data if d["smoothness"] > high_threshold]

    print(
        f"Categorized smoothness: High (≤{low_threshold:.2f}): {len(high_smoothness)}, "
        f"Medium: {len(medium_smoothness)}, Low (>{high_threshold:.2f}): {len(low_smoothness)}"
    )

    return high_smoothness, medium_smoothness, low_smoothness


def extract_features(image_path):
    """Extract features from an image for PCA, resizing to 64x64."""
    img = Image.open(image_path).convert("L")
    # Resize to 64x64 to speed up computation
    img = img.resize((64, 64), Image.NEAREST)
    img_array = np.array(img)

    # Flatten the image and normalize
    features = img_array.flatten() / 255.0

    return features


def cluster_hausdorff_distances(category_data, n_clusters=3):
    """Cluster Hausdorff distances within a category."""
    distances = np.array([d["hausdorff_distance"] for d in category_data]).reshape(
        -1, 1
    )

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(distances)

    # Add cluster information to data
    for i, d in enumerate(category_data):
        d["cluster"] = int(clusters[i])

    # Get cluster centers for reference
    centers = kmeans.cluster_centers_.flatten()

    return category_data, centers


# def perform_pca_visualization(category, category_name):
#     """Perform PCA visualization for a category."""
#     from tqdm import tqdm

#     if not category:
#         print(f"No images in {category_name} category, skipping PCA...")
#         return

#     # Sample up to 100 files proportionally
#     sample_size = min(100, len(category))

#     # Use random sampling without replacement
#     sampled_items = random.sample(category, sample_size)

#     print(
#         f"Sampling {sample_size} files from {len(category)} total files in {category_name} category"
#     )

#     gt_features = []
#     pred_features = []
#     image_ids = []
#     hausdorff_distances = []
#     smoothness_values = []

#     # Use tqdm for progress tracking
#     for item in tqdm(sampled_items, desc=f"Processing {category_name} category"):
#         image_id = item["image_id"]
#         image_ids.append(image_id)
#         hausdorff_distances.append(item["hausdorff_distance"])
#         smoothness_values.append(item["smoothness"])

#         gt_file = os.path.join(ground_truth_path, f"{image_id}_segmentation.png")
#         pred_file = os.path.join(output_mask_path, f"{image_id}.jpg")

#         if os.path.exists(gt_file) and os.path.exists(pred_file):
#             gt_features.append(extract_features(gt_file))
#             pred_features.append(extract_features(pred_file))

#     if not gt_features or not pred_features:
#         print(f"No valid features extracted for {category_name} category")
#         return

#     # Make sure all feature vectors have the same length by padding or truncating
#     max_length = max(len(f) for f in gt_features + pred_features)

#     # Pad or truncate features
#     gt_features_padded = [
#         (
#             np.pad(f, (0, max_length - len(f)), "constant")
#             if len(f) < max_length
#             else f[:max_length]
#         )
#         for f in gt_features
#     ]
#     pred_features_padded = [
#         (
#             np.pad(f, (0, max_length - len(f)), "constant")
#             if len(f) < max_length
#             else f[:max_length]
#         )
#         for f in pred_features
#     ]

#     # Stack features
#     all_features = np.vstack(gt_features_padded + pred_features_padded)

#     # Apply PCA
#     pca = PCA(n_components=2)
#     pca_result = pca.fit_transform(all_features)

#     # Split results back into ground truth and predictions
#     gt_pca = pca_result[: len(gt_features)]
#     pred_pca = pca_result[len(gt_features) :]

#     # ----- 1. Standard PCA visualization (original style) -----
#     plt.figure(figsize=(10, 8))

#     # Ground truth points
#     plt.scatter(
#         gt_pca[:, 0],
#         gt_pca[:, 1],
#         label="Ground Truth",
#         alpha=0.7,
#         marker="o",
#         color="blue",
#     )

#     # Prediction points
#     plt.scatter(
#         pred_pca[:, 0],
#         pred_pca[:, 1],
#         label="Prediction",
#         alpha=0.7,
#         marker="x",
#         color="red",
#     )

#     # # Draw lines connecting corresponding points
#     # for i in range(len(gt_pca)):
#     #     plt.plot(
#     #         [gt_pca[i, 0], pred_pca[i, 0]],
#     #         [gt_pca[i, 1], pred_pca[i, 1]],
#     #         "k-",
#     #         alpha=0.3,
#     #     )

#     plt.title(f"PCA Visualization for {category_name} Smoothness Category")
#     plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2f}%)")
#     plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2f}%)")
#     plt.legend()
#     plt.grid(True, alpha=0.3)

#     # Add variance explained text in top right corner
#     total_var = sum(pca.explained_variance_ratio_)
#     plt.text(
#         0.95,
#         0.95,
#         f"Total variance explained: {total_var:.2f}",
#         transform=plt.gca().transAxes,
#         fontsize=10,
#         verticalalignment="top",
#         horizontalalignment="right",
#         bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
#     )

#     # Save figure
#     fig_path = os.path.join(
#         figs_output_path, f"{class_name}_pca_{category_name.lower()}_smoothness.png"
#     )
#     plt.savefig(fig_path, dpi=300, bbox_inches="tight")
#     print(f"Saved standard PCA visualization to {fig_path}")

#     # ----- 2. PCA with Hausdorff distance color mapping -----
#     plt.figure(figsize=(10, 8))

#     # Ground truth points colored by Hausdorff distance
#     norm = plt.Normalize(min(hausdorff_distances), max(hausdorff_distances))

#     # Ground truth points
#     gt_scatter = plt.scatter(
#         gt_pca[:, 0],
#         gt_pca[:, 1],
#         c=hausdorff_distances,
#         cmap="plasma",
#         alpha=0.7,
#         marker="o",
#         norm=norm,
#     )

#     # Prediction points - use the same colors
#     pred_scatter = plt.scatter(
#         pred_pca[:, 0],
#         pred_pca[:, 1],
#         c=hausdorff_distances,
#         cmap="plasma",
#         alpha=0.7,
#         marker="x",
#         norm=norm,
#     )

#     # Draw lines connecting corresponding points
#     for i in range(len(gt_pca)):
#         plt.plot(
#             [gt_pca[i, 0], pred_pca[i, 0]],
#             [gt_pca[i, 1], pred_pca[i, 1]],
#             "k-",
#             alpha=0.3,
#         )

#     plt.title(
#         f"PCA Visualization for {category_name} Smoothness\nColored by Hausdorff Distance"
#     )
#     plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2f}%)")
#     plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2f}%)")
#     plt.colorbar(gt_scatter, label="Hausdorff Distance")
#     plt.legend(["Ground Truth (o)", "Prediction (x)"])
#     plt.grid(True, alpha=0.3)

#     # Add variance explained text in top right corner
#     plt.text(
#         0.95,
#         0.95,
#         f"Total variance explained: {total_var:.2f}",
#         transform=plt.gca().transAxes,
#         fontsize=10,
#         verticalalignment="top",
#         horizontalalignment="right",
#         bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
#     )

#     # Save figure
#     hd_fig_path = os.path.join(
#         figs_output_path,
#         f"{class_name}_pca_{category_name.lower()}_smoothness_hd_color.png",
#     )
#     plt.savefig(hd_fig_path, dpi=300, bbox_inches="tight")
#     print(f"Saved Hausdorff-colored PCA visualization to {hd_fig_path}")

#     # Create additional visualizations for detailed analysis
#     # Show examples from each category
#     if len(image_ids) > 0:
#         num_examples = min(3, len(image_ids))
#         fig, axes = plt.subplots(num_examples, 2, figsize=(10, 5 * num_examples))

#         if num_examples == 1:
#             axes = np.array([axes])

#         # Sort by Hausdorff distance
#         sorted_indices = np.argsort(hausdorff_distances)

#         for i in range(num_examples):
#             idx = sorted_indices[
#                 i * len(sorted_indices) // num_examples
#             ]  # Evenly distribute examples
#             image_id = image_ids[idx]
#             hd = hausdorff_distances[idx]

#             gt_file = os.path.join(ground_truth_path, f"{image_id}_segmentation.png")
#             pred_file = os.path.join(output_mask_path, f"{image_id}.jpg")

#             axes[i, 0].imshow(np.array(Image.open(gt_file)), cmap="gray")
#             axes[i, 0].set_title(f"Ground Truth: {image_id}\nHD: {hd:.2f}")
#             axes[i, 0].axis("off")

#             axes[i, 1].imshow(np.array(Image.open(pred_file)), cmap="gray")
#             axes[i, 1].set_title(f"Prediction: {image_id}")
#             axes[i, 1].axis("off")

#         plt.suptitle(f"{category_name} Smoothness Category Examples")
#         plt.tight_layout()
#         plt.subplots_adjust(top=0.92)  # Make room for suptitle

#         examples_fig_path = os.path.join(
#             figs_output_path,
#             f"{class_name}_examples_{category_name.lower()}_smoothness.png",
#         )
#         plt.savefig(examples_fig_path, dpi=300, bbox_inches="tight")
#         print(f"Saved example visualizations to {examples_fig_path}")

#     category_with_clusters, cluster_centers = cluster_hausdorff_distances(
#         sampled_items, n_clusters=3
#     )
#     clusters = [item["cluster"] for item in category_with_clusters]

#     # Create additional visualization showing examples from each cluster
#     for cluster_id in set(clusters):
#         cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]

#         if not cluster_indices:
#             continue

#         # Display the top 3 examples from each cluster (or fewer if less than 3)
#         num_examples = min(3, len(cluster_indices))

#         fig, axes = plt.subplots(num_examples, 2, figsize=(10, 5 * num_examples))
#         if num_examples == 1:
#             axes = np.array([axes])

#         # Sort by Hausdorff distance within cluster
#         cluster_items = [(i, hausdorff_distances[i]) for i in cluster_indices]
#         sorted_items = sorted(cluster_items, key=lambda x: x[1])

#         for i in range(num_examples):
#             if i < len(sorted_items):
#                 idx, hd = sorted_items[i]
#                 image_id = image_ids[idx]

#                 gt_file = os.path.join(
#                     ground_truth_path, f"{image_id}_segmentation.png"
#                 )
#                 pred_file = os.path.join(output_mask_path, f"{image_id}.jpg")

#                 axes[i, 0].imshow(np.array(Image.open(gt_file)), cmap="gray")
#                 axes[i, 1].imshow(np.array(Image.open(pred_file)), cmap="gray")

#                 if i == 0:
#                     # axes[i, 0].set_title(f"Ground Truth: {image_id}\nHD: {hd:.2f}")
#                     # axes[i, 1].set_title(f"Prediction: {image_id}")
#                     axes[i, 0].set_title(f"Ground Truth")
#                     axes[i, 1].set_title(f"Prediction")

#                 axes[i, 0].axis("off")
#                 axes[i, 1].axis("off")

#         plt.suptitle(f"{category_name} Smoothness", fontsize=20)
#         # plt.suptitle(
#         #     f"{category_name} Smoothness - Cluster {cluster_id} Examples (HD ≈ {cluster_centers[cluster_id]:.2f})"
#         # )
#         plt.tight_layout()
#         # plt.subplots_adjust(top=0.92)  # Make room for suptitle

#         # Save figure
#         cluster_fig_path = os.path.join(
#             figs_output_path,
#             f"{class_name}_{category_name.lower()}_smoothness_cluster{cluster_id}_examples.png",
#         )
#         plt.savefig(cluster_fig_path, dpi=300, bbox_inches="tight")
#         print(f"Saved cluster {cluster_id} examples to {cluster_fig_path}")


def analyze_hausdorff_by_smoothness(analysis_data, categories):
    """Analyze Hausdorff distance distribution across smoothness categories."""
    category_names = ["High Smoothness", "Medium Smoothness", "Low Smoothness"]

    # Create figure
    # 2:3
    plt.figure(figsize=(6, 9))

    # Create boxplots for each category
    hausdorff_by_category = []
    for category in categories:
        hausdorff_by_category.append([d["hausdorff_distance"] for d in category])

    box = plt.boxplot(hausdorff_by_category, labels=category_names, patch_artist=True)

    # Set colors for boxes
    box_colors = ["lightblue", "lightgreen", "salmon"]
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)

    plt.title(f"Hausdorff Distance Distribution by Smoothness Category", fontsize=20)
    plt.ylabel("Hausdorff Distance")
    plt.grid(True, axis="y", alpha=0.3)

    # Add mean values as text
    for i, category in enumerate(hausdorff_by_category):
        if category:
            mean_val = np.mean(category)
            plt.text(
                i + 1,
                max(category) * 0.9,
                f"Mean: {mean_val:.2f}",
                horizontalalignment="center",
                size="small",
                bbox=dict(facecolor="white", alpha=0.7),
            )

    # Save figure
    fig_path = os.path.join(
        figs_output_path, f"{class_name}_hausdorff_by_smoothness.png"
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved Hausdorff by smoothness analysis to {fig_path}")

    # Create scatter plot of smoothness vs hausdorff distance
    plt.figure(figsize=(12, 9))

    smoothness = [d["smoothness"] for d in analysis_data]
    hausdorff = [d["hausdorff_distance"] for d in analysis_data]

    plt.scatter(smoothness, hausdorff, alpha=0.7)
    plt.title(
        f"Relationship between Mask Smoothness and Hausdorff Distance", fontsize=20
    )
    plt.xlabel("Smoothness (perimeter²/area)")
    plt.ylabel("Hausdorff Distance")
    plt.grid(True, alpha=0.3)

    # Add trendline
    if smoothness and hausdorff:
        z = np.polyfit(smoothness, hausdorff, 1)
        p = np.poly1d(z)
        x_sorted = sorted(smoothness)
        plt.plot(x_sorted, p(x_sorted), "r--", alpha=0.8)

        # Calculate and display correlation
        correlation = np.corrcoef(smoothness, hausdorff)[0, 1]
        plt.text(
            0.05,
            0.95,
            f"Correlation: {correlation:.2f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    # Save figure
    scatter_path = os.path.join(
        figs_output_path, f"{class_name}_smoothness_vs_hausdorff.png"
    )
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    print(f"Saved smoothness vs Hausdorff scatter plot to {scatter_path}")


def main():
    print(f"Starting analysis for {class_name}...")

    # Analyze masks and calculate Hausdorff distances
    analysis_data = analyze_masks()

    # Categorize masks based on ground truth smoothness
    high_smoothness, medium_smoothness, low_smoothness = categorize_by_smoothness(
        analysis_data
    )
    categories = [high_smoothness, medium_smoothness, low_smoothness]

    # Analyze Hausdorff distances across smoothness categories
    analyze_hausdorff_by_smoothness(analysis_data, categories)

    # Perform PCA visualization for each smoothness category
    for category, name in zip(categories, ["High", "Medium", "Low"]):
        perform_pca_visualization(category, name)

    print(f"Analysis complete for {class_name}!")


if __name__ == "__main__":
    main()
