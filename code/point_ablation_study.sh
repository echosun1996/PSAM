#!/bin/bash

# Auto-detect available GPUs
available_gpus=$(nvidia-smi --list-gpus | wc -l)
if [ $available_gpus -eq 0 ]; then
    echo "Warning: No GPUs detected. Using CPU mode."
    gpu_sum=1
    gpu_id=""
else
    gpu_sum=$available_gpus
    # Generate GPU ID list: 0,1,2,3,... for all available GPUs
    gpu_id=$(seq -s, 0 $((available_gpus - 1)))
fi

echo "Detected $available_gpus GPU(s), using GPU IDs: $gpu_id"

# Configuration
img_resize=1024
labeller=42
parent_dir=$(dirname "$(pwd)")
current_user=$(whoami)

# Specify the image filename (without path, e.g., ISIC_0024985.jpg)
# image_filename="ISIC_0024985.jpg"
# image_filename="ISIC_0024563.jpg"
# image_filename="ISIC_0024312.jpg"
# image_filename="ISIC_0024655.jpg"

# image_filename="ISIC_0024398.jpg"
# image_filename="ISIC_0024526.jpg"
# image_filename="ISIC_0024625.jpg"
image_filename="ISIC_0024661.jpg"

image_basename=$(basename "$image_filename" .jpg)

# Output directory for results
output_base_dir="$parent_dir/results/point_ablation"

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
echo "Point Ablation Study for Image: $image_filename"
echo "#########################################################################"

cd $parent_dir/code

# Create output directory
mkdir -p $output_base_dir

# Scenario 1: n-negative-points=4, n-positive-points from 1-10
echo "#########################################################################"
echo "Scenario 1: n-negative-points=4, n-positive-points from 1-10"
echo "#########################################################################"

for n_pos in {1..8}; do
    dir_name="nneg4_npos${n_pos}"
    output_dir="$output_base_dir/$dir_name"
    mask_dir="$output_dir/HAM10000/val/mask"
    pred_mask_path="$mask_dir/$image_basename.jpg"
    
    echo "Processing: n-positive-points=$n_pos, n-negative-points=4"
    echo "Output directory: $output_dir"
    
    # Check if target image already exists
    if [ -f "$pred_mask_path" ]; then
        echo "⏭️  Skipping: Target image already exists at $pred_mask_path"
        echo ""
        continue    
    fi
    
    CUDA_VISIBLE_DEVICES=$gpu_id $torchrun_cmd --nnodes=1 --nproc_per_node=1 --master_port=$(( RANDOM % 49152 + 10000 )) train.py \
        --eval \
        --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth \
        --model-type vit_b \
        --output $output_dir \
        --restore-model $parent_dir/results/PSAM/epoch_10.pth \
        --input $parent_dir/data \
        --input_size $img_resize $img_resize \
        --logfile $output_dir/PSAM_eva.txt \
        --visualize \
        --comments "Point ablation: n-neg=4, n-pos=$n_pos" \
        --labeller $labeller \
        --token-visualisation True \
        --n-positive-points $n_pos \
        --n-negative-points 4 \
        --single-image "$image_filename" \
        --seed 42
    
    echo "✅ Finished: $dir_name"
    echo ""
done

# Scenario 2: n-positive-points=4, n-negative-points from 1-10
echo "#########################################################################"
echo "Scenario 2: n-positive-points=4, n-negative-points from 1-10"
echo "#########################################################################"
cd $parent_dir/code

for n_neg in {1..8}; do
    dir_name="npos4_nneg${n_neg}"
    output_dir="$output_base_dir/$dir_name"
    mask_dir="$output_dir/HAM10000/val/mask"
    pred_mask_path="$mask_dir/$image_basename.jpg"
    
    echo "Processing: n-positive-points=4, n-negative-points=$n_neg"
    echo "Output directory: $output_dir"
    
    # Check if target image already exists
    if [ -f "$pred_mask_path" ]; then
        echo "⏭️  Skipping: Target image already exists at $pred_mask_path"
        echo ""
        continue
    fi

    CUDA_VISIBLE_DEVICES=$gpu_id $torchrun_cmd --nnodes=1 --nproc_per_node=1 --master_port=$(( RANDOM % 49152 + 10000 )) train.py \
        --eval \
        --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth \
        --model-type vit_b \
        --output $output_dir \
        --restore-model $parent_dir/results/PSAM/epoch_10.pth \
        --input $parent_dir/data \
        --input_size $img_resize $img_resize \
        --logfile $output_dir/PSAM_eva.txt \
        --visualize \
        --comments "Point ablation: n-pos=4, n-neg=$n_neg" \
        --labeller $labeller \
        --token-visualisation True \
        --n-positive-points 4 \
        --n-negative-points $n_neg \
        --single-image "$image_filename" \
        --seed 42
    
    echo "✅ Finished: $dir_name"
    echo ""
done

echo "#########################################################################"
echo "Generating combined visualizations..."
echo "#########################################################################"

# Paths for the specific image
original_img_path="$parent_dir/data/HAM10000/input/val/HAM10000_img/$image_filename"
gt_mask_path="$parent_dir/data/HAM10000/input/val/HAM10000_seg/$image_basename"_segmentation.png
points_csv_path="$parent_dir/data/HAM10000/input/HAM10000_val_prompts_${labeller}.csv"

# Check if files exist
if [ ! -f "$original_img_path" ]; then
    echo "Error: Original image not found: $original_img_path"
    exit 1
fi

if [ ! -f "$gt_mask_path" ]; then
    echo "Error: Ground truth mask not found: $gt_mask_path"
    exit 1
fi

if [ ! -f "$points_csv_path" ]; then
    echo "Error: Points CSV not found: $points_csv_path"
    exit 1
fi

cd $parent_dir/code/py_scripts/figs

# Generate combined images for Scenario 1 using point_ablation_study.py
echo "Generating combined images for Scenario 1 using point_ablation_study.py..."
for n_pos in {1..8}; do
    dir_name="nneg4_npos${n_pos}"
    mask_dir="$output_base_dir/$dir_name/HAM10000/val/mask"
    output_dir="$output_base_dir/$dir_name"
    pred_mask_path="$mask_dir/$image_basename.jpg"
    ps_mask_path="$mask_dir/${image_basename}_ps.png"
    ns_mask_path="$mask_dir/${image_basename}_ns.png"
    
    # Check if required files exist
    if [ -f "$pred_mask_path" ] && [ -f "$ps_mask_path" ] && [ -f "$ns_mask_path" ]; then
        echo "Processing $dir_name..."
        python point_ablation_study.py \
            --predict-input "$mask_dir" \
            --intergrated-output "$output_dir" \
            --image "$parent_dir/data/HAM10000/input/val/HAM10000_img" \
            --gt-mask "$parent_dir/data/HAM10000/input/val/HAM10000_seg" \
            --points-csv "$points_csv_path" \
            --single-image "$image_filename" \
            --prompt-type "P_B"
    else
        echo "Warning: Missing files for $dir_name, skipping..."
    fi
done

# Generate combined images for Scenario 2 using point_ablation_study.py
echo "Generating combined images for Scenario 2 using point_ablation_study.py..."
for n_neg in {1..8}; do
    dir_name="npos4_nneg${n_neg}"
    mask_dir="$output_base_dir/$dir_name/HAM10000/val/mask"
    output_dir="$output_base_dir/$dir_name"
    pred_mask_path="$mask_dir/$image_basename.jpg"
    ps_mask_path="$mask_dir/${image_basename}_ps.png"
    ns_mask_path="$mask_dir/${image_basename}_ns.png"
    
    # Check if required files exist
    if [ -f "$pred_mask_path" ] && [ -f "$ps_mask_path" ] && [ -f "$ns_mask_path" ]; then
        echo "Processing $dir_name..."
        python point_ablation_study.py \
            --predict-input "$mask_dir" \
            --intergrated-output "$output_dir" \
            --image "$parent_dir/data/HAM10000/input/val/HAM10000_img" \
            --gt-mask "$parent_dir/data/HAM10000/input/val/HAM10000_seg" \
            --points-csv "$points_csv_path" \
            --single-image "$image_filename" \
            --prompt-type "P_B"
    else
        echo "Warning: Missing files for $dir_name, skipping..."
    fi
done

# Scenario 3: prompt_type=P_B, n-sample-points=4
echo "#########################################################################"
echo "Scenario 3: prompt_type=P_B, n-sample-points=4"
echo "#########################################################################"

cd $parent_dir/code/
dir_name="prompt_P_B_nsample4"
output_dir="$output_base_dir/$dir_name"
mask_dir="$output_dir/HAM10000/val/mask"
pred_mask_path="$mask_dir/$image_basename.jpg"

echo "Processing: prompt_type=P_B, n-sample-points=4"
echo "Output directory: $output_dir"

# Check if target image already exists
if [ ! -f "$pred_mask_path" ]; then
    CUDA_VISIBLE_DEVICES=$gpu_id $torchrun_cmd --nnodes=1 --nproc_per_node=1 --master_port=$(( RANDOM % 49152 + 10000 )) train.py \
        --eval \
        --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth \
        --model-type vit_b \
        --output $output_dir \
        --restore-model $parent_dir/results/PSAM/epoch_10.pth \
        --input $parent_dir/data \
        --input_size $img_resize $img_resize \
        --logfile $output_dir/PSAM_eva.txt \
        --visualize \
        --comments "Prompt ablation: prompt_type=P_B, n-sample-points=4" \
        --labeller $labeller \
        --token-visualisation True \
        --n-sample-points 4 \
        --prompt_type P_B \
        --single-image "$image_filename" \
        --seed 42
    
    echo "✅ Finished: $dir_name"
else
    echo "⏭️  Skipping: Target image already exists at $pred_mask_path"
fi

# Generate visualization for Scenario 3
cd $parent_dir/code

if [ -f "$pred_mask_path" ] && [ -f "$mask_dir/${image_basename}_ps.png" ] && [ -f "$mask_dir/${image_basename}_ns.png" ]; then
    echo "Generating visualization for $dir_name..."
    cd $parent_dir/code/py_scripts/figs
    python point_ablation_study.py \
        --predict-input "$mask_dir" \
        --intergrated-output "$output_dir" \
        --image "$parent_dir/data/HAM10000/input/val/HAM10000_img" \
        --gt-mask "$parent_dir/data/HAM10000/input/val/HAM10000_seg" \
        --points-csv "$parent_dir/data/HAM10000/input/HAM10000_val_prompts_${labeller}.csv" \
        --single-image "$image_filename" \
        --prompt-type "P_B"
fi

# Scenario 4: prompt_type=P, n-sample-points=4
echo "#########################################################################"
echo "Scenario 4: prompt_type=P, n-sample-points=4"
echo "#########################################################################"
cd $parent_dir/code

dir_name="prompt_P_nsample4"
output_dir="$output_base_dir/$dir_name"
mask_dir="$output_dir/HAM10000/val/mask"
pred_mask_path="$mask_dir/$image_basename.jpg"

echo "Processing: prompt_type=P, n-sample-points=4"
echo "Output directory: $output_dir"

# Check if target image already exists
if [ ! -f "$pred_mask_path" ]; then
    CUDA_VISIBLE_DEVICES=$gpu_id $torchrun_cmd --nnodes=1 --nproc_per_node=1 --master_port=$(( RANDOM % 49152 + 10000 )) train.py \
        --eval \
        --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth \
        --model-type vit_b \
        --output $output_dir \
        --restore-model $parent_dir/results/PSAM/epoch_10.pth \
        --input $parent_dir/data \
        --input_size $img_resize $img_resize \
        --logfile $output_dir/PSAM_eva.txt \
        --visualize \
        --comments "Prompt ablation: prompt_type=P, n-sample-points=4" \
        --labeller $labeller \
        --token-visualisation True \
        --n-sample-points 4 \
        --prompt_type P \
        --n-positive-points 4 \
        --n-negative-points 4 \
        --single-image "$image_filename" \
        --seed 42
    
    echo "✅ Finished: $dir_name"
else
    echo "⏭️  Skipping: Target image already exists at $pred_mask_path"
fi

# Generate visualization for Scenario 4
if [ -f "$pred_mask_path" ] && [ -f "$mask_dir/${image_basename}_ps.png" ] && [ -f "$mask_dir/${image_basename}_ns.png" ]; then
    echo "Generating visualization for $dir_name..."
    cd $parent_dir/code/py_scripts/figs
    python point_ablation_study.py \
        --predict-input "$mask_dir" \
        --intergrated-output "$output_dir" \
        --image "$parent_dir/data/HAM10000/input/val/HAM10000_img" \
        --gt-mask "$parent_dir/data/HAM10000/input/val/HAM10000_seg" \
        --points-csv "$parent_dir/data/HAM10000/input/HAM10000_val_prompts_${labeller}.csv" \
        --single-image "$image_filename" \
        --prompt-type "P"
fi

# Scenario 5: prompt_type=B, n-sample-points=4
echo "#########################################################################"
echo "Scenario 5: prompt_type=B, n-sample-points=4"
echo "#########################################################################"
cd $parent_dir/code

dir_name="prompt_B_nsample4"
output_dir="$output_base_dir/$dir_name"
mask_dir="$output_dir/HAM10000/val/mask"
pred_mask_path="$mask_dir/$image_basename.jpg"

echo "Processing: prompt_type=B, n-sample-points=4"
echo "Output directory: $output_dir"

# Check if target image already exists
if [ ! -f "$pred_mask_path" ]; then
    CUDA_VISIBLE_DEVICES=$gpu_id $torchrun_cmd --nnodes=1 --nproc_per_node=1 --master_port=$(( RANDOM % 49152 + 10000 )) train.py \
        --eval \
        --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth \
        --model-type vit_b \
        --output $output_dir \
        --restore-model $parent_dir/results/PSAM/epoch_10.pth \
        --input $parent_dir/data \
        --input_size $img_resize $img_resize \
        --logfile $output_dir/PSAM_eva.txt \
        --visualize \
        --comments "Prompt ablation: prompt_type=B, n-sample-points=4" \
        --labeller $labeller \
        --token-visualisation True \
        --n-sample-points 4 \
        --prompt_type B \
        --n-positive-points 4 \
        --n-negative-points 4 \
        --single-image "$image_filename" \
        --seed 42
    
    echo "✅ Finished: $dir_name"
else
    echo "⏭️  Skipping: Target image already exists at $pred_mask_path"
fi

# Generate visualization for Scenario 5
if [ -f "$pred_mask_path" ] && [ -f "$mask_dir/${image_basename}_ps.png" ] && [ -f "$mask_dir/${image_basename}_ns.png" ]; then
    echo "Generating visualization for $dir_name..."
    cd $parent_dir/code/py_scripts/figs
    python point_ablation_study.py \
        --predict-input "$mask_dir" \
        --intergrated-output "$output_dir" \
        --image "$parent_dir/data/HAM10000/input/val/HAM10000_img" \
        --gt-mask "$parent_dir/data/HAM10000/input/val/HAM10000_seg" \
        --points-csv "$parent_dir/data/HAM10000/input/HAM10000_val_prompts_${labeller}.csv" \
        --single-image "$image_filename" \
        --prompt-type "B"
fi

echo "#########################################################################"
echo "✅ All processing completed!"
echo "Results saved to: $output_base_dir"
echo "#########################################################################"




dir_name="Original"
mask_dir="$parent_dir/results/token_visualisation/HAM10000/val/mask"
output_dir="$output_base_dir/$dir_name"
pred_mask_path="$mask_dir/$image_basename.jpg"
ps_mask_path="$mask_dir/${image_basename}_ps.png"
ns_mask_path="$mask_dir/${image_basename}_ns.png"

# Check if required files exist
if [ -f "$pred_mask_path" ] && [ -f "$ps_mask_path" ] && [ -f "$ns_mask_path" ]; then
    echo "Processing $dir_name..."
    python point_ablation_study.py \
        --predict-input "$mask_dir" \
        --intergrated-output "$output_dir" \
        --image "$parent_dir/data/HAM10000/input/val/HAM10000_img" \
        --gt-mask "$parent_dir/data/HAM10000/input/val/HAM10000_seg" \
        --points-csv "$points_csv_path" \
        --single-image "$image_filename" \
        --prompt-type "P_B"
else
    echo "Warning: Missing files for $dir_name, skipping..."
fi