#!/bin/bash

#########################################################################
# Script: generate_comparison_visualizations.sh
# Purpose: Generate visualization images for a specific HAM10000 image
#          from multiple PSAM result directories
#
# Usage:
#   1. Modify the configuration section below (image_filename, dataset, etc.)
#   2. Run: bash generate_comparison_visualizations.sh
#   3. Or: cd code && bash generate_comparison_visualizations.sh
#
# Output:
#   Visualizations will be saved to: results/comparison_visualizations/
#   Each source directory will have its own subdirectory with the visualization
#########################################################################

# Suspending the script if any command fails
set -e
parent_dir=$(dirname "$(pwd)")
current_user=$(whoami)

# Summons protecting animal
echo "#########################################################################"
echo "        ┏┓   ┏┓+ +                                                       "
echo "       ┏┛┻━━━┛┻┓ + +                                                     "
echo "       ┃       ┃                                                         "
echo "       ┃   ━   ┃ ++ + + +                                                "
echo "       ████━████ ┃+                                                      "
echo "       ┃       ┃ +                                                       "
echo "       ┃   ┻   ┃                                                         "
echo "       ┃       ┃ + +                                                     "
echo "       ┗━┓   ┏━┛                                                         "
echo "         ┃   ┃                                                           "
echo "         ┃   ┃ + + + +                                                   "
echo "         ┃   ┃    Code is far away from bug with the animal protecting   "
echo "         ┃   ┃ +     神兽保佑,代码无bug                                    "
echo "         ┃   ┃                                                           "
echo "         ┃   ┃                                                           "
echo "         ┃   ┃  +                                                        "
echo "         ┃    ┗━━━┓ + +                                                  "
echo "                  ┃                                                      "
echo "         ┃        ┣┓                                                     "
echo "         ┃        ┏┛                                                     "
echo "         ┗┓┓┏━┳┓┏┛ + + + +                                               "
echo "          ┃┫┫ ┃┫┫                                                        "
echo "          ┗┻┛ ┗┻┛+ + + +                                                 "
echo "#########################################################################"

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

# Check data folder
if [ ! -d "$parent_dir/data" ]; then
    echo "Error: 'data' folder not found in the parent directory."
    exit 1
fi

############################
# Configuration           ##
############################
# Image filename to process (e.g., ISIC_0024312.jpg)
image_filename="ISIC_0024312.jpg"

# Dataset split: val or test
dataset="val"

# Color map for visualization (e.g., Oranges, Blues, Reds, etc.)
color="Oranges"

# Output directory for generated visualizations
output_dir="$parent_dir/results/comparison_visualizations"

# Source directories to process (relative to results/)
source_dirs=(
    "PSAM_cp_1_P_B"
    "PSAM_cp_2_P_B"
    "PSAM_cp_3_P_B"
    "PSAM_cp_4_B"
    "PSAM_cp_4_P"
    "PSAM_cp_4_P_B"
    "PSAM_cp_5_P_B"
    "PSAM_cp_6_P_B"
    "PSAM_cp_7_P_B"
    "PSAM_cp_8_P_B"
    "PSAM_cp_9_P_B"
    "PSAM_cp_10_P_B"
)

############################
# Generate Visualizations ##
############################
echo "#########################################################################"
echo "Generating comparison visualizations for HAM10000 image"
echo "#########################################################################"
echo "Image filename: $image_filename"
echo "Dataset: $dataset"
echo "Color map: $color"
echo "Output directory: $output_dir"
echo "Source directories: ${#source_dirs[@]}"
echo "#########################################################################"

cd $parent_dir/code/py_scripts

# Run the Python script with source directories as arguments
python generate_comparison_visualizations.py \
    --image-filename "$image_filename" \
    --dataset "$dataset" \
    --output-dir "$output_dir" \
    --color "$color" \
    --parent-dir "$parent_dir" \
    --source-dirs "${source_dirs[@]}"

echo "#########################################################################"
echo "✅ Visualization generation completed!"
echo "Output directory: $output_dir"
echo "#########################################################################"

