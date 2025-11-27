#!/bin/bash

# Script to find best case images with 4pos4neg optimal condition
# Usage:
#   ./find_best_case.sh                    # Process all images
#   ./find_best_case.sh ISIC_0024661.jpg   # Process single image
#   ./find_best_case.sh --image ISIC_0024661.jpg  # Process single image (explicit)

# Auto-detect available GPUs
available_gpus=$(nvidia-smi --list-gpus | wc -l)
if [ $available_gpus -eq 0 ]; then
    echo "Warning: No GPUs detected. Using CPU mode."
    gpu_sum=1
    gpu_id=""
    max_parallel_jobs=1
else
    gpu_sum=$available_gpus
    # Generate GPU ID list: 0,1,2,3,... for all available GPUs
    gpu_id=$(seq -s, 0 $((available_gpus - 1)))
    # Set max parallel jobs: use 2 jobs per GPU for better utilization
    max_parallel_jobs=$((available_gpus * 2))
fi

echo "Detected $available_gpus GPU(s), using GPU IDs: $gpu_id"
echo "Max parallel jobs: $max_parallel_jobs"

# Configuration
img_resize=1024
labeller=42
parent_dir=$(dirname "$(pwd)")
current_user=$(whoami)

# Parse command line arguments for single image processing
SINGLE_IMAGE=""
if [ "$1" = "--image" ] || [ "$1" = "-i" ]; then
    if [ -z "$2" ]; then
        echo "Error: --image requires an image filename (e.g., ISIC_0024661.jpg)"
        exit 1
    fi
    SINGLE_IMAGE="$2"
    echo "Single image mode: Processing only $SINGLE_IMAGE"
    # Set max parallel jobs to 1 for single image mode
    max_parallel_jobs=1
elif [ -n "$1" ] && [[ ! "$1" =~ ^- ]]; then
    # If first argument doesn't start with -, treat it as image name
    SINGLE_IMAGE="$1"
    echo "Single image mode: Processing only $SINGLE_IMAGE"
    # Set max parallel jobs to 1 for single image mode
    max_parallel_jobs=1
fi

# Output directory for results
output_base_dir="$parent_dir/results/find_best_case"
temp_results_dir="$output_base_dir/temp_results"
metrics_dir="$output_base_dir/metrics"

# Load anaconda and environment
if [ -f "$HOME/anaconda3/bin/activate" ]; then
    . "$HOME/anaconda3/bin/activate"
elif [ -f "/opt/anaconda3/bin/activate" ]; then
    . "/opt/anaconda3/bin/activate"
elif [ -f "$HOME/miniconda3/bin/activate" ]; then
    . "$HOME/miniconda3/bin/activate"
fi

# Activate conda environment if specified
if [ -n "$CONDA_ENV_NAME" ]; then
    conda activate "$CONDA_ENV_NAME"
elif command -v conda &> /dev/null; then
    echo "Using default conda environment. Set CONDA_ENV_NAME to specify a different environment."
else
    echo "Conda not found. Installing dependencies with pip..."
    pip install -r requirements.txt
fi

# Use torchrun (should be available in PATH if conda is activated)
torchrun_cmd="torchrun"

echo "#########################################################################"
echo "Finding best case image with 4pos4neg optimal condition"
echo "#########################################################################"

cd $parent_dir/code

# Output file for found images
found_images_file="$output_base_dir/found_images.txt"
lock_file="$output_base_dir/.write_lock"

# Create output directories
mkdir -p $output_base_dir
mkdir -p $temp_results_dir
mkdir -p $metrics_dir

# Initialize the found images file
echo "# Images meeting criteria (4pos4neg optimal)" > "$found_images_file"
echo "# Generated on: $(date)" >> "$found_images_file"
echo "" >> "$found_images_file"

# Function to safely append to found images file
append_to_file() {
    local filename=$1
    (
        flock -x 200
        echo "$filename" >> "$found_images_file"
    ) 200>"$lock_file"
}

# Get list of all images
image_dir="$parent_dir/data/HAM10000/input/val/HAM10000_img"
gt_mask_dir="$parent_dir/data/HAM10000/input/val/HAM10000_seg"

if [ ! -d "$image_dir" ]; then
    echo "Error: Image directory not found: $image_dir"
    exit 1
fi

# Get image files based on SINGLE_IMAGE variable
if [ -n "$SINGLE_IMAGE" ]; then
    # Single image mode: process only the specified image
    # Remove .jpg extension if present
    image_basename=$(basename "$SINGLE_IMAGE" .jpg)
    image_filename="${image_basename}.jpg"
    image_path="$image_dir/$image_filename"
    
    if [ ! -f "$image_path" ]; then
        echo "Error: Image file not found: $image_path"
        exit 1
    fi
    
    image_files=("$image_filename")
    total_images=1
    echo "Single image mode: Processing only $image_filename"
else
    # Normal mode: process all images
    image_files=($(ls "$image_dir"/*.jpg 2>/dev/null | xargs -n1 basename))
    total_images=${#image_files[@]}
    
    if [ $total_images -eq 0 ]; then
        echo "Error: No images found in $image_dir"
        exit 1
    fi
    
    echo "Found $total_images images to process"
fi
echo ""

# Python script to calculate Dice score for a single image
python_calc_script="$temp_results_dir/calc_dice.py"
cat > "$python_calc_script" << 'PYTHON_EOF'
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

class Dice(nn.Module):
    def __init__(self):
        super(Dice, self).__init__()
    
    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return dice

def calculate_dice(predicted_path, gt_path, img_size=(1024, 1024)):
    try:
        pred_img = Image.open(predicted_path).convert("L")
        gt_img = Image.open(gt_path).convert("L")
        
        pred_array = np.array(pred_img)
        gt_array = np.array(gt_img)
        
        # Convert to binary
        if not set(np.unique(pred_array)).issubset({0, 1}) or set(np.unique(pred_array)).issubset({0}):
            pred_array = (pred_array > 127).astype(np.uint8)
        
        if not set(np.unique(gt_array)).issubset({0, 1}) or set(np.unique(gt_array)).issubset({0}):
            gt_array = (gt_array > 127).astype(np.uint8)
        
        # Resize if needed
        if pred_array.shape != img_size:
            pred_img = pred_img.resize(img_size, Image.NEAREST)
            pred_array = np.array(pred_img)
            pred_array = (pred_array > 127).astype(np.uint8)
        
        if gt_array.shape != img_size:
            gt_img = gt_img.resize(img_size, Image.NEAREST)
            gt_array = np.array(gt_img)
            gt_array = (gt_array > 127).astype(np.uint8)
        
        pred_tensor = torch.tensor(pred_array, dtype=torch.float32)
        gt_tensor = torch.tensor(gt_array, dtype=torch.float32)
        
        dice = Dice()
        dice_score = dice(pred_tensor, gt_tensor)
        return dice_score.item()
    except Exception as e:
        print(f"Error calculating dice: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python calc_dice.py <predicted_path> <gt_path>", file=sys.stderr)
        sys.exit(1)
    
    predicted_path = sys.argv[1]
    gt_path = sys.argv[2]
    dice_score = calculate_dice(predicted_path, gt_path)
    if dice_score is not None:
        print(f"{dice_score:.6f}")
    else:
        sys.exit(1)
PYTHON_EOF

chmod +x "$python_calc_script"

# Python script to check criteria incrementally
python_check_script="$temp_results_dir/check_criteria.py"
cat > "$python_check_script" << 'PYTHON_CHECK_EOF'
import sys
import json

def check_incremental(dice_4pos4neg, new_key, new_score, existing_scores):
    """
    Incrementally check if a new score meets the criteria:
    1. All scores should be lower than 4pos4neg
    2. No monotonicity requirement between 2 and 3, they just need to be less than 4
    Returns: (passed, message)
    """
    # Condition 1: New score should be lower than 4pos4neg
    if new_score >= dice_4pos4neg:
        return False, f"{new_key} ({new_score:.6f}) >= 4_4 ({dice_4pos4neg:.6f})"
    
    # For 2, 3, 5, 6: only check they are less than 4pos4neg (already checked above)
    # No monotonicity requirement between 2 and 3
    
    return True, "OK"

if __name__ == "__main__":
    if len(sys.argv) != 5:  # script name + 4 arguments
        print(f"Error: Expected 5 arguments (including script name), got {len(sys.argv)}", file=sys.stderr)
        print(f"Arguments received: {sys.argv}", file=sys.stderr)
        print("Usage: python check_criteria.py <dice_4pos4neg> <new_key> <new_score> <existing_scores_json>", file=sys.stderr)
        sys.exit(1)
    
    try:
        dice_4pos4neg = float(sys.argv[1])
        new_key = sys.argv[2]
        new_score = float(sys.argv[3])
        existing_scores_json = sys.argv[4]
        existing_scores = json.loads(existing_scores_json) if existing_scores_json and existing_scores_json != "{}" else {}
        
        passed, message = check_incremental(dice_4pos4neg, new_key, new_score, existing_scores)
        print(message)
        sys.exit(0 if passed else 1)
    except Exception as e:
        print(f"Error processing arguments: {e}", file=sys.stderr)
        print(f"Arguments: {sys.argv}", file=sys.stderr)
        sys.exit(1)
PYTHON_CHECK_EOF

# Function to get checkpoint based on point count
get_checkpoint() {
    local n_points=$1
    case $n_points in
        2|6)
            echo "$parent_dir/results/PSAM/epoch_30.pth"
            ;;
        3|5)
            echo "$parent_dir/results/PSAM/epoch_30.pth"
            ;;
        4)
            echo "$parent_dir/results/PSAM/epoch_30.pth"
            ;;
        *)
            echo "$parent_dir/results/PSAM/epoch_30.pth"  # Default
            ;;
    esac
}

# Function to check if an image meets the criteria (incremental check and run experiments)
check_criteria() {
    local image_filename=$1
    local assigned_gpu=$2
    local image_basename=$(basename "$image_filename" .jpg)
    
    # Dictionary to store dice scores: key="npos_nneg", value=dice_score
    declare -A dice_scores
    
    # Step 1: First calculate 4pos4neg as baseline
    echo "[GPU $assigned_gpu] [$image_basename] Step 1: Calculating baseline 4pos4neg..."
    dir_name="npos4_nneg4"
    output_dir="$temp_results_dir/$image_basename/$dir_name"
    mask_dir="$output_dir/HAM10000/val/mask"
    pred_mask_path="$mask_dir/$image_basename.jpg"
    gt_mask_path="$gt_mask_dir/${image_basename}_segmentation.png"
    
    if [ ! -f "$pred_mask_path" ]; then
        echo "[GPU $assigned_gpu] [$image_basename] Missing prediction for 4pos4neg, skipping..."
        return 1
    fi
    
    dice_4pos4neg=$(python "$python_calc_script" "$pred_mask_path" "$gt_mask_path" 2>/dev/null)
    if [ -z "$dice_4pos4neg" ]; then
        echo "[GPU $assigned_gpu] [$image_basename] Failed to calculate dice for 4pos4neg, skipping..."
        return 1
    fi
    
    dice_scores["4_4"]=$dice_4pos4neg
    echo "[GPU $assigned_gpu] [$image_basename] 4pos4neg: Dice=$dice_4pos4neg"
    
    # Step 2: Test Scenario 1: n-negative-points=4, n-positive-points from 2-6
    # Order: 3, 2, 5, 6 (closest to 4 first, then check monotonicity)
    echo "[GPU $assigned_gpu] [$image_basename] Step 2: Testing Scenario 1 (n-neg=4, n-pos from 2-6)..."
    for n_pos in 3 2 5 6; do
        dir_name="nneg4_npos${n_pos}"
        output_dir="$temp_results_dir/$image_basename/$dir_name"
        mask_dir="$output_dir/HAM10000/val/mask"
        pred_mask_path="$mask_dir/$image_basename.jpg"
        
        # Run experiment if prediction doesn't exist
        if [ ! -f "$pred_mask_path" ]; then
            echo "[GPU $assigned_gpu] [$image_basename] Running experiment: n-pos=$n_pos, n-neg=4"
            checkpoint=$(get_checkpoint $n_pos)
            CUDA_VISIBLE_DEVICES=$assigned_gpu $torchrun_cmd --nnodes=1 --nproc_per_node=1 --master_port=$(( RANDOM % 49152 + 10000 )) train.py \
                --eval \
                --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth \
                --model-type vit_b \
                --output $output_dir \
                --restore-model $checkpoint \
                --input $parent_dir/data \
                --input_size $img_resize $img_resize \
                --logfile $output_dir/PSAM_eva.txt \
                --visualize \
                --comments "Find best case: n-neg=4, n-pos=$n_pos" \
                --labeller $labeller \
                --token-visualisation True \
                --n-positive-points $n_pos \
                --n-negative-points 4 \
                --single-image "$image_filename" \
                --seed 42 > /dev/null 2>&1
            
            if [ ! -f "$pred_mask_path" ]; then
                echo "[GPU $assigned_gpu] [$image_basename] Failed to generate prediction for n-pos=$n_pos, skipping..."
                return 1
            fi
        fi
        
        # Calculate dice score
        dice_score=$(python "$python_calc_script" "$pred_mask_path" "$gt_mask_path" 2>/dev/null)
        if [ -z "$dice_score" ]; then
            echo "[GPU $assigned_gpu] [$image_basename] Failed to calculate dice for n-pos=$n_pos, skipping..."
            return 1
        fi
        
        echo "[GPU $assigned_gpu] [$image_basename] n-pos=$n_pos, n-neg=4: Dice=$dice_score"
        
        # Convert existing scores to JSON
        dice_json="{"
        first=true
        for key in "${!dice_scores[@]}"; do
            if [ "$first" = true ]; then
                first=false
            else
                dice_json+=", "
            fi
            dice_json+="\"$key\": ${dice_scores[$key]}"
        done
        dice_json+="}"
        
        # Ensure dice_json is not empty (should at least have 4_4)
        if [ "$dice_json" = "{}" ] && [ ${#dice_scores[@]} -eq 0 ]; then
            echo "[GPU $assigned_gpu] [$image_basename] Error: dice_scores is empty, cannot check criteria"
            return 1
        fi
        
        # Verify Python script exists
        if [ ! -f "$python_check_script" ]; then
            echo "[GPU $assigned_gpu] [$image_basename] Error: Python check script not found: $python_check_script"
            return 1
        fi
        
        # Check incrementally - if fails, return immediately
        result=$(python "$python_check_script" "$dice_4pos4neg" "${n_pos}_4" "$dice_score" "$dice_json" 2>&1)
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "[GPU $assigned_gpu] [$image_basename] ❌ Failed at n-pos=$n_pos: $result"
            echo "[GPU $assigned_gpu] [$image_basename] Debug: dice_4pos4neg=$dice_4pos4neg, key=${n_pos}_4, score=$dice_score"
            echo "[GPU $assigned_gpu] [$image_basename] Debug: json length=${#dice_json}, json=$dice_json"
            return 1
        fi
        
        dice_scores["${n_pos}_4"]=$dice_score
    done
    
    # Step 3: Test Scenario 2: n-positive-points=4, n-negative-points from 2-6
    # Order: 3, 2, 5, 6 (closest to 4 first, then check monotonicity)
    echo "[GPU $assigned_gpu] [$image_basename] Step 3: Testing Scenario 2 (n-pos=4, n-neg from 2-6)..."
    for n_neg in 3 2 5 6; do
        dir_name="npos4_nneg${n_neg}"
        output_dir="$temp_results_dir/$image_basename/$dir_name"
        mask_dir="$output_dir/HAM10000/val/mask"
        pred_mask_path="$mask_dir/$image_basename.jpg"
        
        # Run experiment if prediction doesn't exist
        if [ ! -f "$pred_mask_path" ]; then
            echo "[GPU $assigned_gpu] [$image_basename] Running experiment: n-pos=4, n-neg=$n_neg"
            checkpoint=$(get_checkpoint $n_neg)
            CUDA_VISIBLE_DEVICES=$assigned_gpu $torchrun_cmd --nnodes=1 --nproc_per_node=1 --master_port=$(( RANDOM % 49152 + 10000 )) train.py \
                --eval \
                --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth \
                --model-type vit_b \
                --output $output_dir \
                --restore-model $checkpoint \
                --input $parent_dir/data \
                --input_size $img_resize $img_resize \
                --logfile $output_dir/PSAM_eva.txt \
                --visualize \
                --comments "Find best case: n-pos=4, n-neg=$n_neg" \
                --labeller $labeller \
                --token-visualisation True \
                --n-positive-points 4 \
                --n-negative-points $n_neg \
                --single-image "$image_filename" \
                --seed 42 > /dev/null 2>&1
            
            if [ ! -f "$pred_mask_path" ]; then
                echo "[GPU $assigned_gpu] [$image_basename] Failed to generate prediction for n-neg=$n_neg, skipping..."
                return 1
            fi
        fi
        
        # Calculate dice score
        dice_score=$(python "$python_calc_script" "$pred_mask_path" "$gt_mask_path" 2>/dev/null)
        if [ -z "$dice_score" ]; then
            echo "[GPU $assigned_gpu] [$image_basename] Failed to calculate dice for n-neg=$n_neg, skipping..."
            return 1
        fi
        
        echo "[GPU $assigned_gpu] [$image_basename] n-pos=4, n-neg=$n_neg: Dice=$dice_score"
        
        # Convert existing scores to JSON
        dice_json="{"
        first=true
        for key in "${!dice_scores[@]}"; do
            if [ "$first" = true ]; then
                first=false
            else
                dice_json+=", "
            fi
            dice_json+="\"$key\": ${dice_scores[$key]}"
        done
        dice_json+="}"
        
        # Ensure dice_json is not empty (should at least have 4_4)
        if [ "$dice_json" = "{}" ] && [ ${#dice_scores[@]} -eq 0 ]; then
            echo "[GPU $assigned_gpu] [$image_basename] Error: dice_scores is empty, cannot check criteria"
            return 1
        fi
        
        # Verify Python script exists
        if [ ! -f "$python_check_script" ]; then
            echo "[GPU $assigned_gpu] [$image_basename] Error: Python check script not found: $python_check_script"
            return 1
        fi
        
        # Check incrementally - if fails, return immediately
        result=$(python "$python_check_script" "$dice_4pos4neg" "4_${n_neg}" "$dice_score" "$dice_json" 2>&1)
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "[GPU $assigned_gpu] [$image_basename] ❌ Failed at n-neg=$n_neg: $result"
            echo "[GPU $assigned_gpu] [$image_basename] Debug: dice_4pos4neg=$dice_4pos4neg, key=4_${n_neg}, score=$dice_score"
            echo "[GPU $assigned_gpu] [$image_basename] Debug: json length=${#dice_json}, json=$dice_json"
            return 1
        fi
        
        dice_scores["4_${n_neg}"]=$dice_score
    done
    
    echo "[GPU $assigned_gpu] [$image_basename] ✅ All conditions met!"
    return 0
}

# Function to process a single image
process_image() {
    local image_filename=$1
    local assigned_gpu=$2
    local image_basename=$(basename "$image_filename" .jpg)
    
    echo "[GPU $assigned_gpu] #########################################################################"
    echo "[GPU $assigned_gpu] Processing: $image_filename"
    echo "[GPU $assigned_gpu] #########################################################################"
    
    # Check if ground truth exists
    local gt_mask_path="$gt_mask_dir/${image_basename}_segmentation.png"
    if [ ! -f "$gt_mask_path" ]; then
        echo "[GPU $assigned_gpu] [$image_basename] ⏭️  Skipping: Ground truth mask not found"
        return 1
    fi
    
    # Step 1: First run 4pos4neg as baseline (if not exists)
    local dir_name="npos4_nneg4"
    local output_dir="$temp_results_dir/$image_basename/$dir_name"
    local mask_dir="$output_dir/HAM10000/val/mask"
    local pred_mask_path="$mask_dir/$image_basename.jpg"
    
    if [ ! -f "$pred_mask_path" ]; then
        echo "[GPU $assigned_gpu] [$image_basename] Running baseline experiment: n-positive-points=4, n-negative-points=4"
        checkpoint=$(get_checkpoint 4)
        CUDA_VISIBLE_DEVICES=$assigned_gpu $torchrun_cmd --nnodes=1 --nproc_per_node=1 --master_port=$(( RANDOM % 49152 + 10000 )) train.py \
            --eval \
            --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth \
            --model-type vit_b \
            --output $output_dir \
            --restore-model $checkpoint \
            --input $parent_dir/data \
            --input_size $img_resize $img_resize \
            --logfile $output_dir/PSAM_eva.txt \
            --visualize \
            --comments "Find best case: n-pos=4, n-neg=4" \
            --labeller $labeller \
            --token-visualisation True \
            --n-positive-points 4 \
            --n-negative-points 4 \
            --single-image "$image_filename" \
            --seed 42 > /dev/null 2>&1
    fi
    
    # Check criteria incrementally (this will run remaining experiments and check on the fly)
    if ! check_criteria "$image_filename" "$assigned_gpu"; then
        echo "[GPU $assigned_gpu] [$image_basename] ❌ Does not meet criteria"
        return 1
    fi
    
    # If we reach here, the image meets all criteria
    echo "[GPU $assigned_gpu] [$image_basename] 🎉 FOUND MATCHING IMAGE: $image_filename"
    append_to_file "$image_filename"
    return 0
}

# Function to generate visualizations for a found image
generate_visualizations() {
    local image_filename=$1
    local image_basename=$(basename "$image_filename" .jpg)
    
    echo "#########################################################################"
    echo "Generating visualizations for: $image_filename"
    echo "#########################################################################"
    
    # Paths for the specific image
    local original_img_path="$parent_dir/data/HAM10000/input/val/HAM10000_img/$image_filename"
    local gt_mask_path="$parent_dir/data/HAM10000/input/val/HAM10000_seg/${image_basename}_segmentation.png"
    local points_csv_path="$parent_dir/data/HAM10000/input/HAM10000_val_prompts_${labeller}.csv"
    
    # Check if files exist
    if [ ! -f "$original_img_path" ]; then
        echo "Error: Original image not found: $original_img_path"
        return 1
    fi
    
    if [ ! -f "$gt_mask_path" ]; then
        echo "Error: Ground truth mask not found: $gt_mask_path"
        return 1
    fi
    
    if [ ! -f "$points_csv_path" ]; then
        echo "Error: Points CSV not found: $points_csv_path"
        return 1
    fi
    
    cd $parent_dir/code/py_scripts/figs
    
    # Generate combined images for Scenario 1: n-negative-points=4, n-positive-points from 2-6
    echo "Generating combined images for Scenario 1 (n-neg=4, n-pos from 2-6)..."
    for n_pos in 2 3 4 5 6; do
        local dir_name="nneg4_npos${n_pos}"
        local mask_dir="$temp_results_dir/$image_basename/$dir_name/HAM10000/val/mask"
        local output_dir="$temp_results_dir/$image_basename/$dir_name"
        local pred_mask_path="$mask_dir/$image_basename.jpg"
        local ps_mask_path="$mask_dir/${image_basename}_ps.png"
        local ns_mask_path="$mask_dir/${image_basename}_ns.png"
        
        # Check if required files exist
        if [ -f "$pred_mask_path" ] && [ -f "$ps_mask_path" ] && [ -f "$ns_mask_path" ]; then
            echo "  Processing $dir_name..."
            python point_ablation_study.py \
                --predict-input "$mask_dir" \
                --intergrated-output "$output_dir" \
                --image "$parent_dir/data/HAM10000/input/val/HAM10000_img" \
                --gt-mask "$parent_dir/data/HAM10000/input/val/HAM10000_seg" \
                --points-csv "$points_csv_path" \
                --single-image "$image_filename" \
                --prompt-type "P_B"
        else
            echo "  Warning: Missing files for $dir_name, skipping..."
        fi
    done
    
    # Generate combined images for Scenario 2: n-positive-points=4, n-negative-points from 2-6
    echo "Generating combined images for Scenario 2 (n-pos=4, n-neg from 2-6)..."
    for n_neg in 2 3 4 5 6; do
        local dir_name="npos4_nneg${n_neg}"
        local mask_dir="$temp_results_dir/$image_basename/$dir_name/HAM10000/val/mask"
        local output_dir="$temp_results_dir/$image_basename/$dir_name"
        local pred_mask_path="$mask_dir/$image_basename.jpg"
        local ps_mask_path="$mask_dir/${image_basename}_ps.png"
        local ns_mask_path="$mask_dir/${image_basename}_ns.png"
        
        # Check if required files exist
        if [ -f "$pred_mask_path" ] && [ -f "$ps_mask_path" ] && [ -f "$ns_mask_path" ]; then
            echo "  Processing $dir_name..."
            python point_ablation_study.py \
                --predict-input "$mask_dir" \
                --intergrated-output "$output_dir" \
                --image "$parent_dir/data/HAM10000/input/val/HAM10000_img" \
                --gt-mask "$parent_dir/data/HAM10000/input/val/HAM10000_seg" \
                --points-csv "$points_csv_path" \
                --single-image "$image_filename" \
                --prompt-type "P_B"
        else
            echo "  Warning: Missing files for $dir_name, skipping..."
        fi
    done
    
    echo "✅ Visualizations generated for $image_filename"
    echo ""
}

# Export necessary variables and functions for parallel processing
export parent_dir
export temp_results_dir
export gt_mask_dir
export torchrun_cmd
export img_resize
export labeller
export python_calc_script
export python_check_script
export found_images_file
export lock_file
export -f get_checkpoint
export -f check_criteria
export -f process_image

# Process images in parallel
echo "#########################################################################"
echo "Starting parallel processing of $total_images images"
echo "Max parallel jobs: $max_parallel_jobs"
echo "#########################################################################"
echo ""

found_images=()
image_count=0
running_jobs=0
declare -A pids
declare -A gpu_assignments

# Function to wait for jobs and assign GPU
wait_for_slot() {
    while [ $running_jobs -ge $max_parallel_jobs ]; do
        # Check for completed jobs
        for pid in "${!pids[@]}"; do
            if ! kill -0 $pid 2>/dev/null; then
                # Job completed, remove from tracking
                unset pids[$pid]
                running_jobs=$((running_jobs - 1))
            fi
        done
        sleep 0.5
    done
}

# Function to assign GPU to a job (round-robin)
assign_gpu() {
    local job_idx=$1
    if [ $available_gpus -eq 0 ]; then
        echo ""
    else
        echo $((job_idx % available_gpus))
    fi
}

# Process each image in parallel
for image_filename in "${image_files[@]}"; do
    image_count=$((image_count + 1))
    
    # Wait for available slot
    wait_for_slot
    
    # Assign GPU (round-robin)
    assigned_gpu=$(assign_gpu $image_count)
    
    # Start processing in background
    (
        # Redefine append_to_file function in subshell (flock may not work across processes)
        append_to_file() {
            local filename=$1
            (
                flock -x 200
                echo "$filename" >> "$found_images_file"
            ) 200>"$lock_file"
        }
        
        process_image "$image_filename" "$assigned_gpu"
    ) &
    
    pid=$!
    pids[$pid]=1
    running_jobs=$((running_jobs + 1))
    
    echo "[Main] Started processing $image_filename on GPU $assigned_gpu (PID: $pid, Running: $running_jobs/$max_parallel_jobs)"
done

# Wait for all remaining jobs to complete
echo ""
echo "[Main] Waiting for all jobs to complete..."
while [ ${#pids[@]} -gt 0 ]; do
    for pid in "${!pids[@]}"; do
        if ! kill -0 $pid 2>/dev/null; then
            unset pids[$pid]
            running_jobs=$((running_jobs - 1))
        fi
    done
    if [ ${#pids[@]} -gt 0 ]; then
        sleep 1
    fi
done

echo "[Main] All jobs completed!"
echo ""

# Read found images from file
if [ -f "$found_images_file" ]; then
    while IFS= read -r line; do
        if [[ ! "$line" =~ ^# ]] && [ -n "$line" ]; then
            found_images+=("$line")
        fi
    done < "$found_images_file"
fi

# Print summary
echo "#########################################################################"
echo "Summary"
echo "#########################################################################"
echo "Total images processed: $total_images"
echo "Images meeting criteria: ${#found_images[@]}"
echo ""

if [ ${#found_images[@]} -gt 0 ]; then
    echo "✅ Found matching images:"
    for img in "${found_images[@]}"; do
        echo "   - $img"
    done
    echo ""
    echo "📄 Found images saved to: $found_images_file"
    echo ""
    
    # Generate visualizations for found images
    echo "#########################################################################"
    echo "Generating visualizations for found images..."
    echo "#########################################################################"
    echo ""
    
    for img in "${found_images[@]}"; do
        generate_visualizations "$img"
    done
    
    echo "#########################################################################"
    echo "✅ All visualizations generated!"
    echo "#########################################################################"
else
    echo "❌ No images found that meet all criteria"
fi

echo ""
echo "Results saved to: $temp_results_dir"
echo "#########################################################################"

