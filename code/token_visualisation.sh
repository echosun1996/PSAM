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

# Resizing image
split_and_resize=false
img_resize=1024
ham_train_rate=0.8
ham_val_rate=0.1
max_epoch_num=32
# Prompts labeller No.
labeller=42


scribble_type=""
# scribble_type="LinePAndLineN"
# scribble_type="XPAndLineN" # X scribble as positive and Line scribble as negative

# Output folder name in results folder
output_folder_name="PSAM"
if [ "$scribble_type" = "" ]; then
    echo "Training model without scribble..."
else
    echo "Training model with scribble:" $scribble_type
    output_folder_name="PSAM_"$scribble_type
fi

check_dataset=false


# Training model
evaluate_model=false
sp_den_mask_integration=false
uncertain_vis=true

# Regenerate token visualisation for PSAM_cp_* directories
regenerate_cp_dirs=false

add_mask=false
filename_var="ISIC_0024985.jpg"
filename_var="ISIC_0027571.jpg"
filename_var="ISIC_0029334.jpg"
# filename_var="ISIC_0029804.jpg"


wandb_comment="max_epoch_num:32. Original training. Three specific positive points and four negative points, box+point+noisy mask on specific point, add 4 negative points"


if [ "$scribble_type" = "LinePAndLineN" ]; then
    wandb_comment="Use Line as positive and negative prompt; "$wandb_comment
elif [ "$scribble_type" = "XPAndLineN" ]; then
    wandb_comment="Use X as positive and Line as negative prompt; "$wandb_comment
fi

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

# relogin wandb (use environment variable WANDB_API_KEY if set)
if [ -n "$WANDB_API_KEY" ]; then
    wandb login --relogin "$WANDB_API_KEY"
else
    echo "Warning: WANDB_API_KEY not set. Skipping wandb login."
    echo "To use wandb, set WANDB_API_KEY environment variable."
fi


# Check data folder
if [ ! -d "$parent_dir/data" ]; then
    echo "Error: 'data' folder not found in the parent directory."
    exit 1
fi

############################
# Download all dataset    ##
############################
echo "Downloing original datasets..."
# Create folder for each dataset
mkdir -p $parent_dir/data/HAM10000/original
mkdir -p $parent_dir/data/ISIC2016/original
mkdir -p $parent_dir/data/ISIC2017/original
mkdir -p $parent_dir/data/PH2/original
mkdir -p $parent_dir/data/Atlas/original
mkdir -p $parent_dir/data/Dermofit/original

# HAM10000
cd $parent_dir/data/HAM10000/original
## HAM10000 images
file_count=-1
if [ -d "HAM10000_images" ]; then
    file_count=$(ls -1 HAM10000_images | wc -l)
fi
if [ $file_count -ne 10015 ]; then
    rm -rf HAM10000_images
    wget -O "HAM10000_images_part_1.zip" "https://example.com/placeholder/download"
    unzip -d HAM10000_images HAM10000_images_part_1.zip 
    wget -O "HAM10000_images_part_2.zip" "https://example.com/placeholder/download"
    unzip -d HAM10000_images HAM10000_images_part_2.zip 
else
    echo "[√] HAM10000 Images check pass, skipping download."
fi
## HAM10000 GroundTruth
file_count=-1
if [ -d "HAM10000_GroundTruth" ]; then
    file_count=$(ls -1 HAM10000_GroundTruth | wc -l)
fi
if [ $file_count -ne 10015 ]; then
    rm -rf HAM10000_GroundTruth
    wget -O "HAM10000_segmentations_lesion_tschandl.zip" "https://example.com/placeholder/download"
    unzip -j -d HAM10000_GroundTruth HAM10000_segmentations_lesion_tschandl.zip 
else
    echo "[√] HAM10000 GroundTruth check pass, skipping download."
fi

# ISIC 2016
cd $parent_dir/data/ISIC2016/original
## ISIC2016 Training images
file_count=-1
if [ -d "ISIC_2016_Training/ISBI2016_ISIC_Part1_Training_Data" ]; then
    file_count=$(ls -1 ISIC_2016_Training/ISBI2016_ISIC_Part1_Training_Data | wc -l)
fi
if [ $file_count -ne 900 ]; then
    rm -rf ISIC_2016_Training
    wget https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_Data.zip
    unzip -d ISIC_2016_Training ISBI2016_ISIC_Part1_Training_Data.zip
else
    echo "[√] ISIC2016 Training Images check pass, skipping download."
fi
## ISIC2016 Training GroundTruth
file_count=-1
if [ -d "ISIC_2016_Training/ISBI2016_ISIC_Part1_Training_GroundTruth" ]; then
    file_count=$(ls -1 ISIC_2016_Training/ISBI2016_ISIC_Part1_Training_GroundTruth | wc -l)
fi
if [ $file_count -ne 900 ]; then
    rm -rf ISIC_2016_Training
    wget https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_GroundTruth.zip
    unzip -d ISIC_2016_Training ISBI2016_ISIC_Part1_Training_GroundTruth.zip
else
    echo "[√] ISIC2016 Training GroundTruth check pass, skipping download."
fi
## ISIC2016 Test images
file_count=-1
if [ -d "ISIC_2016_Test/ISBI2016_ISIC_Part1_Test_Data" ]; then
    file_count=$(ls -1 ISIC_2016_Test/ISBI2016_ISIC_Part1_Test_Data | wc -l)
fi
if [ $file_count -ne 379 ]; then
    rm -rf ISIC_2016_Test
    wget https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_Data.zip
    unzip -d ISIC_2016_Test ISBI2016_ISIC_Part1_Test_Data.zip
else
    echo "[√] ISIC2016 Test Images check pass, skipping download."
fi  
## ISIC2016 Test GroundTruth
file_count=-1
if [ -d "ISIC_2016_Test/ISBI2016_ISIC_Part1_Test_GroundTruth" ]; then
    file_count=$(ls -1 ISIC_2016_Test/ISBI2016_ISIC_Part1_Test_GroundTruth | wc -l)
fi
if [ $file_count -ne 379 ]; then
    rm -rf ISIC_2016_Test
    wget https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_GroundTruth.zip
    unzip -d ISIC_2016_Test ISBI2016_ISIC_Part1_Test_GroundTruth.zip
else
    echo "[√] ISIC2016 Test GroundTruth check pass, skipping download."
fi


# ISIC 2017
cd $parent_dir/data/ISIC2017/original
## ISIC2017 Training images
file_count=-1
if [ -d "ISIC_2017_Training/ISIC-2017_Training_Data" ]; then
    file_count=$(ls -1 ISIC_2017_Training/ISIC-2017_Training_Data | wc -l)
fi
if [ $file_count -ne 4001 ]; then
    rm -rf ISIC_2017_Training
    wget https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip
    unzip -d ISIC_2017_Training ISIC-2017_Training_Data.zip
else
    echo "[√] ISIC2017 Training Images check pass, skipping download."
fi  
## ISIC2017 Training Ground Truth
file_count=-1
if [ -d "ISIC_2017_Training/ISIC-2017_Training_Part1_GroundTruth" ]; then
    file_count=$(ls -1 ISIC_2017_Training/ISIC-2017_Training_Part1_GroundTruth | wc -l)
fi
if [ $file_count -ne 2000 ]; then
    rm -rf ISIC_2017_Training
    wget https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part1_GroundTruth.zip
    unzip -d ISIC_2017_Training ISIC-2017_Training_Part1_GroundTruth.zip
else
    echo "[√] ISIC2017 Training GroundTruth check pass, skipping download."
fi
## ISIC2017 Validation images
file_count=-1
if [ -d "ISIC_2017_Validation/ISIC-2017_Validation_Data" ]; then
    file_count=$(ls -1 ISIC_2017_Validation/ISIC-2017_Validation_Data | wc -l)
fi
if [ $file_count -ne 301 ]; then
    rm -rf ISIC_2017_Validation
    wget https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Data.zip
    unzip -d ISIC_2017_Validation ISIC-2017_Validation_Data.zip
else
    echo "[√] ISIC2017 Validation Images check pass, skipping download."
fi  
## ISIC2017 Validation Ground Truth
file_count=-1
if [ -d "ISIC_2017_Validation/ISIC-2017_Validation_Part1_GroundTruth" ]; then
    file_count=$(ls -1 ISIC_2017_Validation/ISIC-2017_Validation_Part1_GroundTruth | wc -l)
fi
if [ $file_count -ne 150 ]; then
    rm -rf ISIC_2017_Validation
    wget https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Part1_GroundTruth.zip
    unzip -d ISIC_2017_Validation ISIC-2017_Validation_Part1_GroundTruth.zip
else
    echo "[√] ISIC2017 Validation GroundTruth check pass, skipping download."
fi
## ISIC2017 Test images
file_count=-1
if [ -d "ISIC_2017_Test/ISIC-2017_Test_v2_Data" ]; then
    file_count=$(ls -1 ISIC_2017_Test/ISIC-2017_Test_v2_Data | wc -l)
fi
if [ $file_count -ne 1201 ]; then
    rm -rf ISIC_2017_Test
    wget https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Data.zip
    unzip -d ISIC_2017_Test ISIC-2017_Test_v2_Data.zip
else
    echo "[√] ISIC2017 Test Images check pass, skipping download."
fi  
## ISIC2017 Test Ground Truth
file_count=-1
if [ -d "ISIC_2017_Test/ISIC-2017_Test_v2_Part1_GroundTruth" ]; then
    file_count=$(ls -1 ISIC_2017_Test/ISIC-2017_Test_v2_Part1_GroundTruth | wc -l)
fi
if [ $file_count -ne 600 ]; then
    rm -rf ISIC_2017_Test
    wget https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part1_GroundTruth.zip
    unzip -d ISIC_2017_Test ISIC-2017_Test_v2_Part1_GroundTruth.zip
else
    echo "[√] ISIC2017 Test GroundTruth exists, skipping download."
fi

# PH2
cd $parent_dir/data/PH2/original
rm -rf PH2_temp
## PH2 images and ground truth
file_count=-1
if [ -d "PH2_Data" ]; then
    file_count=$(ls -1 PH2_Data | wc -l)
fi
if [ $file_count -ne 200 ]; then
    rm -rf PH2_GroundTruth
    rm -rf PH2_Data
    wget -c -O "PH2Dataset.rar" "https://example.com/placeholder/ph2_dataset.rar"
    unrar x PH2Dataset.rar PH2_temp/
    python $parent_dir/code/py_scripts/preprocessing/PH2_prepare.py --input-dir PH2_temp --output-dir $parent_dir/data/PH2/original
    rm -rf PH2_temp
else
    echo "[√] PH2 check pass, skipping download."
fi

# Atlas
cd $parent_dir/data/Atlas/original
## Atlas images and ground truth
file_count=-1
if [ -d "Atlas_Data" ]; then
    file_count=$(ls -1 Atlas_Data | wc -l)
fi
if [ $file_count -ne 960 ]; then
    rm -rf Atlas_GroundTruth
    rm -rf Atlas_Data
    wget -c -O "AtlasDataset.zip" "https://example.com/placeholder/atlas_dataset.zip"
    unzip -j AtlasDataset.zip 'AtlasDataset/Atlas_GroundTruth/*' -d ./Atlas_GroundTruth/ -x '*.DS_Store'
    unzip -j AtlasDataset.zip 'AtlasDataset/Atlas_Data/*' -d ./Atlas_Data/ -x '*.DS_Store'
else
    echo "[√] Atlas check pass, skipping download."
fi

# Dermofit
cd $parent_dir/data/Dermofit/original
groundtruth_dir="$parent_dir/data/Dermofit/original/Dermofit_GroundTruth"
data_dir="$parent_dir/data/Dermofit/original/Dermofit_Data"
## Dermofit images and ground truth
mkdir -p "$groundtruth_dir"
mkdir -p "$data_dir"
file_count=-1
if [ -d "Dermofit_Data" ]; then
    file_count=$(ls -1 Dermofit_Data | wc -l)
fi
if [ $file_count -ne 1300 ]; then
    rm -rf Dermofit_GroundTruth
    rm -rf Dermofit_Data
    wget -c -O "DermofitDataset.zip" "https://example.com/placeholder/dermofit_dataset.zip"
    unzip -j DermofitDataset.zip -d ./DermofitDataset/ -x '*.DS_Store'
    cd ./DermofitDataset/
    find . -name "*.zip" -exec unzip -j {} -d ./ \;
    find . -name "*.zip" -delete

    mkdir -p "$groundtruth_dir"
    mkdir -p "$data_dir"

    find . -type f -name "*mask.png" | while read file; do
        new_file="${file%mask.png}_segmentation.png"
        mv "$file" "$groundtruth_dir/$(basename "$new_file")"
    done

    # 查找并移动其余的 .png 文件，并修改扩展名为 .jpg
    find . -type f -name "*.png" ! -name "*mask.png" | while read file; do
        new_file="$data_dir/$(basename "${file%.png}.jpg")"
        mv "$file" "$new_file"
    done

else
    echo "[√] Dermofit check pass, skipping download."
fi
echo "#########################################################################"

############################
# Split and resize         #
############################
if [ "$split_and_resize" = true ]; then
    echo "Splitting and resizing HAM10000..."
    echo "* trianning rate: $ham_train_rate"
    echo "* validation rate: $ham_val_rate"
    echo "* resizing image to $img_resize x $img_resize"
    cd $parent_dir/code
    rm -rf $parent_dir/data/HAM10000/input
    python py_scripts/preprocessing/HAM_split_and_resize.py --input-path $parent_dir/data/HAM10000/original/  --output-path $parent_dir/data/HAM10000/input/ --image-resize $img_resize $img_resize --train-rate $ham_train_rate --val-rate $ham_val_rate
    echo "[√] HAM10000 split and resize done."

    echo "Resizing ISIC2016..."
    rm -rf $parent_dir/data/ISIC2016/input
    python py_scripts/preprocessing/resize_images.py --input-path $parent_dir/data/ISIC2016/original/ISIC_2016_Training/ISBI2016_ISIC_Part1_Training_Data --output-path $parent_dir/data/ISIC2016/input/train/ISIC2016_img --image-resize $img_resize $img_resize
    python py_scripts/preprocessing/resize_images.py --input-path $parent_dir/data/ISIC2016/original/ISIC_2016_Training/ISBI2016_ISIC_Part1_Training_GroundTruth --output-path $parent_dir/data/ISIC2016/input/train/ISIC2016_seg --image-resize $img_resize $img_resize --is-seg True
    echo "[√] ISIC2016 training resize done."
    python py_scripts/preprocessing/resize_images.py --input-path $parent_dir/data/ISIC2016/original/ISIC_2016_Test/ISBI2016_ISIC_Part1_Test_Data --output-path $parent_dir/data/ISIC2016/input/test/ISIC2016_img --image-resize $img_resize $img_resize
    python py_scripts/preprocessing/resize_images.py --input-path $parent_dir/data/ISIC2016/original/ISIC_2016_Test/ISBI2016_ISIC_Part1_Test_GroundTruth --output-path $parent_dir/data/ISIC2016/input/test/ISIC2016_seg --image-resize $img_resize $img_resize --is-seg True
    echo "[√] ISIC2016 test resize done."

    echo "Resizing ISIC2017..."
    rm -rf $parent_dir/data/ISIC2017/input
    python py_scripts/preprocessing/resize_images.py --input-path $parent_dir/data/ISIC2017/original/ISIC_2017_Training/ISIC-2017_Training_Data --output-path $parent_dir/data/ISIC2017/input/train/ISIC2017_img --image-resize $img_resize $img_resize
    python py_scripts/preprocessing/resize_images.py --input-path $parent_dir/data/ISIC2017/original/ISIC_2017_Training/ISIC-2017_Training_Part1_GroundTruth --output-path $parent_dir/data/ISIC2017/input/train/ISIC2017_seg --image-resize $img_resize $img_resize --is-seg True
    echo "[√] ISIC2017 training resize done."
    python py_scripts/preprocessing/resize_images.py --input-path $parent_dir/data/ISIC2017/original/ISIC_2017_Validation/ISIC-2017_Validation_Data --output-path $parent_dir/data/ISIC2017/input/val/ISIC2017_img --image-resize $img_resize $img_resize
    python py_scripts/preprocessing/resize_images.py --input-path $parent_dir/data/ISIC2017/original/ISIC_2017_Validation/ISIC-2017_Validation_Part1_GroundTruth --output-path $parent_dir/data/ISIC2017/input/val/ISIC2017_seg --image-resize $img_resize $img_resize --is-seg True
    echo "[√] ISIC2017 val resize done."
    python py_scripts/preprocessing/resize_images.py --input-path $parent_dir/data/ISIC2017/original/ISIC_2017_Test/ISIC-2017_Test_v2_Data --output-path $parent_dir/data/ISIC2017/input/test/ISIC2017_img --image-resize $img_resize $img_resize
    python py_scripts/preprocessing/resize_images.py --input-path $parent_dir/data/ISIC2017/original/ISIC_2017_Test/ISIC-2017_Test_v2_Part1_GroundTruth --output-path $parent_dir/data/ISIC2017/input/test/ISIC2017_seg --image-resize $img_resize $img_resize --is-seg True
    echo "[√] ISIC2017 test resize done."

    echo "Resizing PH2..."
    rm -rf $parent_dir/data/PH2/input
    python py_scripts/preprocessing/resize_images.py --input-path $parent_dir/data/PH2/original/PH2_Data --output-path $parent_dir/data/PH2/input/test/PH2_img --image-resize $img_resize $img_resize
    python py_scripts/preprocessing/resize_images.py --input-path $parent_dir/data/PH2/original/PH2_GroundTruth --output-path $parent_dir/data/PH2/input/test/PH2_seg --image-resize $img_resize $img_resize --is-seg True
    echo "[√] PH2 test resize done."

    echo "Resizing Atlas..."
    rm -rf $parent_dir/data/Atlas/input
    python py_scripts/preprocessing/resize_images.py --input-path $parent_dir/data/Atlas/original/Atlas_Data --output-path $parent_dir/data/Atlas/input/test/Atlas_img --image-resize $img_resize $img_resize
    python py_scripts/preprocessing/resize_images.py --input-path $parent_dir/data/Atlas/original/Atlas_GroundTruth --output-path $parent_dir/data/Atlas/input/test/Atlas_seg --image-resize $img_resize $img_resize --is-seg True
    echo "[√] Atlas test resize done."

    echo "Resizing Dermofit..."
    rm -rf $parent_dir/data/Dermofit/input
    python py_scripts/preprocessing/resize_images.py --input-path $parent_dir/data/Dermofit/original/Dermofit_Data --output-path $parent_dir/data/Dermofit/input/test/Dermofit_img --image-resize $img_resize $img_resize
    python py_scripts/preprocessing/resize_images.py --input-path $parent_dir/data/Dermofit/original/Dermofit_GroundTruth --output-path $parent_dir/data/Dermofit/input/test/Dermofit_seg --image-resize $img_resize $img_resize --is-seg True
    echo "[√] Dermofit test resize done."
    echo "#########################################################################"
fi


############################
# Check resize dataset    ##
############################
if [ "$check_dataset" = true ]; then
    echo "Checking image size..."
    cd $parent_dir/code
    python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_train_img --data-path $parent_dir/data/HAM10000/input/train/HAM10000_img/ --image-size $img_resize $img_resize --image-num 8012
    python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_train_seg --data-path $parent_dir/data/HAM10000/input/train/HAM10000_seg/ --image-size $img_resize $img_resize --image-num 8012

    python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_val_img --data-path $parent_dir/data/HAM10000/input/val/HAM10000_img/ --image-size $img_resize $img_resize --image-num 1001
    python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_val_seg --data-path $parent_dir/data/HAM10000/input/val/HAM10000_seg/ --image-size $img_resize $img_resize --image-num 1001

    python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_test_img --data-path $parent_dir/data/HAM10000/input/test/HAM10000_img/ --image-size $img_resize $img_resize --image-num 1002
    python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_test_seg --data-path $parent_dir/data/HAM10000/input/test/HAM10000_seg/ --image-size $img_resize $img_resize --image-num 1002

    python py_scripts/preprocessing/check_image_size.py --data-name ISIC2016_train_img --data-path $parent_dir/data/ISIC2016/input/train/ISIC2016_img/ --image-size $img_resize $img_resize --image-num 900
    python py_scripts/preprocessing/check_image_size.py --data-name ISIC2016_train_seg --data-path $parent_dir/data/ISIC2016/input/train/ISIC2016_seg/ --image-size $img_resize $img_resize --image-num 900

    python py_scripts/preprocessing/check_image_size.py --data-name ISIC2016_test_img --data-path $parent_dir/data/ISIC2016/input/test/ISIC2016_img/ --image-size $img_resize $img_resize --image-num 379
    python py_scripts/preprocessing/check_image_size.py --data-name ISIC2016_test_seg --data-path $parent_dir/data/ISIC2016/input/test/ISIC2016_seg/ --image-size $img_resize $img_resize --image-num 379

    python py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_train_img --data-path $parent_dir/data/ISIC2017/input/train/ISIC2017_img/ --image-size $img_resize $img_resize --image-num 2000
    python py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_train_seg --data-path $parent_dir/data/ISIC2017/input/train/ISIC2017_seg/ --image-size $img_resize $img_resize --image-num 2000

    python py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_val_val --data-path $parent_dir/data/ISIC2017/input/val/ISIC2017_img/ --image-size $img_resize $img_resize --image-num 150
    python py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_val_val --data-path $parent_dir/data/ISIC2017/input/val/ISIC2017_seg/ --image-size $img_resize $img_resize --image-num 150

    python py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_test_img --data-path $parent_dir/data/ISIC2017/input/test/ISIC2017_img/ --image-size $img_resize $img_resize --image-num 600
    python py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_test_seg --data-path $parent_dir/data/ISIC2017/input/test/ISIC2017_seg/ --image-size $img_resize $img_resize --image-num 600

    python py_scripts/preprocessing/check_image_size.py --data-name PH2_img --data-path $parent_dir/data/PH2/input/test/PH2_img/ --image-size $img_resize $img_resize --image-num 200
    python py_scripts/preprocessing/check_image_size.py --data-name PH2_seg --data-path $parent_dir/data/PH2/input/test/PH2_seg/ --image-size $img_resize $img_resize --image-num 200

    python py_scripts/preprocessing/check_image_size.py --data-name Atlas_img --data-path $parent_dir/data/Atlas/input/test/Atlas_img/ --image-size $img_resize $img_resize --image-num 960
    python py_scripts/preprocessing/check_image_size.py --data-name Atlas_seg --data-path $parent_dir/data/Atlas/input/test/Atlas_seg/ --image-size $img_resize $img_resize --image-num 960

    python py_scripts/preprocessing/check_image_size.py --data-name Dermofit_img --data-path $parent_dir/data/Dermofit/input/test/Dermofit_img/ --image-size $img_resize $img_resize --image-num 1300
    python py_scripts/preprocessing/check_image_size.py --data-name Dermofit_seg --data-path $parent_dir/data/Dermofit/input/test/Dermofit_seg/ --image-size $img_resize $img_resize --image-num 1300
    echo "✅ Dataset check pass."
    echo "#########################################################################"
fi

############################
# Check pretrained model  ##
############################
echo "Checking pretrained model..."
cd $parent_dir/code
checkpoint_file="./pretrained_checkpoint/sam_vit_b_01ec64.pth"
download_url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
checkpoint_dir="$(dirname "$checkpoint_file")"
if [ ! -d "$checkpoint_dir" ]; then
    mkdir -p "$checkpoint_dir"
fi
if [ ! -f "$checkpoint_file" ]; then
    echo "$checkpoint_file not found, downloading.."
    wget -c -O "$checkpoint_file" "$download_url"
fi
checkpoint_file="./pretrained_checkpoint/sam_vit_b_maskdecoder.pth"
download_url="https://example.com/placeholder/sam_checkpoint.pth"
checkpoint_dir="$(dirname "$checkpoint_file")"
if [ ! -f "$checkpoint_file" ]; then
    echo "$checkpoint_file not found, downloading.."
    wget -O "$checkpoint_file" "$download_url"
fi
echo "✅ Pretrained model check pass."
echo "#########################################################################"

############################
# Evaluate the model      ##
############################
if [ "$evaluate_model" = true ]; then
    echo "Evaluate the model..."
    cd $parent_dir/code
    
    if [ "$regenerate_cp_dirs" = true ]; then
        # Regenerate token visualisation for PSAM_cp_* directories
        echo "#########################################################################"
        echo "Regenerating token visualisation for PSAM_cp_* directories"
        echo "#########################################################################"
        
        # Use torchrun (should be available in PATH if conda is activated)
        torchrun_cmd="torchrun"
        
        # Define the list of directories and their corresponding n_sample_points
        declare -A cp_dirs=(
            # ["PSAM_cp_1_P_B"]="1"
            ["PSAM_cp_2_P_B"]="2"
            # ["PSAM_cp_3_P_B"]="3"
            # ["PSAM_cp_4_B"]="4"
            # ["PSAM_cp_4_P"]="4"
            # ["PSAM_cp_4_P_B"]="4"
            # ["PSAM_cp_5_P_B"]="5"
            # ["PSAM_cp_6_P_B"]="6"
            # ["PSAM_cp_7_P_B"]="7"
            # ["PSAM_cp_8_P_B"]="8"
            # ["PSAM_cp_9_P_B"]="9"
            # ["PSAM_cp_10_P_B"]="10"
        )
        
        # Determine prompt_type based on directory name
        get_prompt_type() {
            local dir_name=$1
            if [[ "$dir_name" == *"_B" ]] && [[ "$dir_name" != *"_P_B" ]]; then
                echo "B"
            elif [[ "$dir_name" == *"_P" ]] && [[ "$dir_name" != *"_P_B" ]]; then
                echo "P"
            else
                echo "P_B"
            fi
        }
        
        for dir_name in "${!cp_dirs[@]}"; do
            n_sample_points="${cp_dirs[$dir_name]}"
            prompt_type=$(get_prompt_type "$dir_name")
            
            echo "#########################################################################"
            echo "Processing: $dir_name"
            echo "  n_sample_points: $n_sample_points"
            echo "  prompt_type: $prompt_type"
            echo "#########################################################################"
            
            # Only process HAM10000 dataset
            if [ "$scribble_type" = "" ]; then
                # Use all available GPUs for parallel processing
                CUDA_VISIBLE_DEVICES=$gpu_id $torchrun_cmd --nnodes=1 --nproc_per_node=$gpu_sum --master_port=$(( RANDOM % 49152 + 10000 )) train.py --eval --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth --model-type vit_b --output $parent_dir/results/$dir_name --restore-model $parent_dir/results/PSAM/epoch_16.pth --input $parent_dir/data --input_size $img_resize $img_resize --logfile $parent_dir/results/$dir_name/PSAM_eva.txt --visualize  --comments "$wandb_comment" --labeller $labeller --token-visualisation True --n-sample-points $n_sample_points --prompt_type $prompt_type
            else
                echo "Invalid scribble_type: $scribble_type"
                exit 1
            fi
            
            echo "✅ Finished processing: $dir_name"
            echo ""
        done
        
        echo "#########################################################################"
        echo "✅ All PSAM_cp_* directories processed!"
        echo "#########################################################################"
    else
        # Original behavior: evaluate to token_visualisation directory
        # rm -rf $parent_dir/results/$output_folder_name/HAM10000
        # rm -rf $parent_dir/results/$output_folder_name/ISIC2016
        # rm -rf $parent_dir/results/$output_folder_name/ISIC2017
        # rm -rf $parent_dir/results/$output_folder_name/Atlas
        # rm -rf $parent_dir/results/$output_folder_name/Dermofit
        rm -rf $parent_dir/results/token_visualisation

        # CUDA_VISIBLE_DEVICES=$gpu_id torchrun  --master_port=$(( RANDOM % 49152 + 10000 )) --nproc_per_node=$gpu_sum train.py --eval --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth --model-type vit_b --output $parent_dir/results/PSAM --restore-model $parent_dir/results/PSAM/epoch_20.pth --input $parent_dir/data --input_size $img_resize $img_resize --logfile $parent_dir/results/PSAM/PSAM_eva.txt --visualize  --comments "$wandb_comment" --labeller $labeller
        if [ "$scribble_type" = "" ]; then
            torchrun_cmd="torchrun"
            CUDA_VISIBLE_DEVICES=$gpu_id $torchrun_cmd --nnodes=1 --nproc_per_node=$gpu_sum  --master_port=$(( RANDOM % 49152 + 10000 )) train.py --eval --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth --model-type vit_b --output $parent_dir/results/token_visualisation --restore-model $parent_dir/results/PSAM/epoch_16.pth --input $parent_dir/data --input_size $img_resize $img_resize --logfile $parent_dir/results/token_visualisation/PSAM_eva.txt --visualize  --comments "$wandb_comment" --labeller $labeller --token-visualisation True
        else
            echo "Invalid scribble_type: $scribble_type"
            exit 1
        fi
        
        echo "✅ Evaluate finished."
        echo "#########################################################################"
    fi
fi



############################
# integration sp_den_mask_integration 
############################
if [ "$sp_den_mask_integration" = true ]; then
    echo "Evaluate the model..."
    cd $parent_dir/code/py_scripts/figs

    echo "Predicting HAM10000 val..."
    rm -rf $parent_dir/results/token_visualisation/HAM10000/val/integrated
    python sp_den_mask_integration.py --predict-input $parent_dir/results/token_visualisation/HAM10000/val/mask --intergrated-output $parent_dir/results/token_visualisation/HAM10000/val/integrated --image $parent_dir/data/HAM10000/input/val/HAM10000_img --points-csv $parent_dir/data/HAM10000/input/HAM10000_val_prompts_42.csv
    
    echo "Predicting HAM10000 test..."
    rm -rf $parent_dir/results/token_visualisation/HAM10000/test/integrated
    python sp_den_mask_integration.py --predict-input $parent_dir/results/token_visualisation/HAM10000/test/mask --intergrated-output $parent_dir/results/token_visualisation/HAM10000/test/integrated --image $parent_dir/data/HAM10000/input/test/HAM10000_img --points-csv $parent_dir/data/HAM10000/input/HAM10000_test_prompts_42.csv

    echo "Predicting ISIC2016..."
    rm -rf $parent_dir/results/token_visualisation/ISIC2016/test/integrated
    python sp_den_mask_integration.py --predict-input $parent_dir/results/token_visualisation/ISIC2016/test/mask --intergrated-output $parent_dir/results/token_visualisation/ISIC2016/test/integrated --image $parent_dir/data/ISIC2016/input/test/ISIC2016_img --points-csv $parent_dir/data/ISIC2016/input/ISIC2016_test_prompts_42.csv

    echo "Predicting ISIC2017..."
    rm -rf $parent_dir/results/token_visualisation/ISIC2017/test/integrated
    python sp_den_mask_integration.py --predict-input $parent_dir/results/token_visualisation/ISIC2017/test/mask --intergrated-output $parent_dir/results/token_visualisation/ISIC2017/test/integrated --image $parent_dir/data/ISIC2017/input/test/ISIC2017_img --points-csv $parent_dir/data/ISIC2017/input/ISIC2017_test_prompts_42.csv

    echo "Predicting PH2..."
    rm -rf $parent_dir/results/token_visualisation/PH2/test/integrated
    python sp_den_mask_integration.py --predict-input $parent_dir/results/token_visualisation/PH2/test/mask --intergrated-output $parent_dir/results/token_visualisation/PH2/test/integrated --image $parent_dir/data/PH2/input/test/PH2_img --points-csv $parent_dir/data/PH2/input/PH2_test_prompts_42.csv

    echo "Predicting AtlasZoomIn10..."
    rm -rf $parent_dir/results/token_visualisation/AtlasZoomIn10/test/integrated
    python sp_den_mask_integration.py --predict-input $parent_dir/results/token_visualisation/AtlasZoomIn10/test/mask --intergrated-output $parent_dir/results/token_visualisation/AtlasZoomIn10/test/integrated --image $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_img --points-csv $parent_dir/data/AtlasZoomIn10/input/AtlasZoomIn10_test_prompts_42.csv
    
    echo "Predicting Dermofit..."
    rm -rf $parent_dir/results/token_visualisation/Dermofit/test/integrated
    python sp_den_mask_integration.py --predict-input $parent_dir/results/token_visualisation/Dermofit/test/mask --intergrated-output $parent_dir/results/token_visualisation/Dermofit/test/integrated --image $parent_dir/data/Dermofit/input/test/Dermofit_img --points-csv $parent_dir/data/Dermofit/input/Dermofit_test_prompts_42.csv
    
    echo "✅ Evaluate finished."
    echo "#########################################################################"
fi


############################
# integration uncertain_vis       
############################
# Define the list of colormap names you want to use

if [ "$uncertain_vis" = true ]; then
    echo "Evaluate the model..."
    color_maps=("Accent" "Accent_r" "Blues" "Blues_r" "BrBG" "BrBG_r" "BuGn" "BuGn_r" "BuPu" "BuPu_r" "CMRmap" "CMRmap_r" "Dark2" "Dark2_r" "GnBu" "GnBu_r" "Greens" "Greens_r" "Greys" "Greys_r" "OrRd" "OrRd_r" "Oranges" "Oranges_r" "PRGn" "PRGn_r" "Paired" "Paired_r" "Pastel1" "Pastel1_r" "Pastel2" "Pastel2_r" "PiYG" "PiYG_r" "PuBu" "PuBuGn" "PuBuGn_r" "PuBu_r" "PuOr" "PuOr_r" "PuRd" "PuRd_r" "Purples" "Purples_r" "RdBu" "RdBu_r" "RdGy" "RdGy_r" "RdPu" "RdPu_r" "RdYlBu" "RdYlBu_r" "RdYlGn" "RdYlGn_r" "Reds" "Reds_r" "Set1" "Set1_r" "Set2" "Set2_r" "Set3" "Set3_r" "Spectral" "Spectral_r" "Wistia" "Wistia_r" "YlGn" "YlGnBu" "YlGnBu_r" "YlGn_r" "YlOrBr" "YlOrBr_r" "YlOrRd" "YlOrRd_r" "afmhot" "afmhot_r" "autumn" "autumn_r" "binary" "binary_r" "bone" "bone_r" "brg" "brg_r" "bwr" "bwr_r" "cividis" "cividis_r" "cool" "cool_r" "coolwarm" "coolwarm_r" "copper" "copper_r" "cubehelix" "cubehelix_r" "flag" "flag_r" "gist_earth" "gist_earth_r" "gist_gray" "gist_gray_r" "gist_heat" "gist_heat_r" "gist_ncar" "gist_ncar_r" "gist_rainbow" "gist_rainbow_r" "gist_stern" "gist_stern_r" "gist_yarg" "gist_yarg_r" "gnuplot" "gnuplot2" "gnuplot2_r" "gnuplot_r" "gray" "gray_r" "hot" "hot_r" "hsv" "hsv_r" "inferno" "inferno_r" "jet" "jet_r" "magma" "magma_r" "nipy_spectral" "nipy_spectral_r" "ocean" "ocean_r" "pink" "pink_r" "plasma" "plasma_r" "prism" "prism_r" "rainbow" "rainbow_r" "seismic" "seismic_r" "spring" "spring_r" "summer" "summer_r" "tab10" "tab10_r" "tab20" "tab20_r" "tab20b" "tab20b_r" "tab20c" "tab20c_r" "terrain" "terrain_r" "turbo" "turbo_r" "twilight" "twilight_r" "twilight_shifted" "twilight_shifted_r" "viridis" "viridis_r" "winter" "winter_r")
    cd $parent_dir/code/py_scripts/figs
    color="Oranges"

    
    echo "Predicting HAM10000 test..."
    rm -rf $parent_dir/results/token_visualisation/HAM10000/val/uncertain_vis
    python uncertain_vis.py --predict-input $parent_dir/results/token_visualisation/HAM10000/val/mask --intergrated-output $parent_dir/results/token_visualisation/HAM10000/val/uncertain_vis --image $parent_dir/data/HAM10000/input/val/HAM10000_img --points-csv $parent_dir/data/HAM10000/input/HAM10000_val_prompts_42.csv --color $color --revised True


    # echo "Predicting HAM10000 val..."
    # rm -rf $parent_dir/results/token_visualisation/HAM10000/val/uncertain_vis
    # python uncertain_vis.py --predict-input $parent_dir/results/token_visualisation/HAM10000/val/mask --intergrated-output $parent_dir/results/token_visualisation/HAM10000/val/uncertain_vis --image $parent_dir/data/HAM10000/input/val/HAM10000_img --points-csv $parent_dir/data/HAM10000/input/HAM10000_val_prompts_42.csv --color $color
    
    # echo "Predicting HAM10000 test..."
    # rm -rf $parent_dir/results/token_visualisation/HAM10000/test/uncertain_vis
    # python uncertain_vis.py --predict-input $parent_dir/results/token_visualisation/HAM10000/test/mask --intergrated-output $parent_dir/results/token_visualisation/HAM10000/test/uncertain_vis --image $parent_dir/data/HAM10000/input/test/HAM10000_img --points-csv $parent_dir/data/HAM10000/input/HAM10000_test_prompts_42.csv --color $color

    # echo "Predicting ISIC2016..."
    # rm -rf $parent_dir/results/token_visualisation/ISIC2016/test/uncertain_vis
    # python uncertain_vis.py --predict-input $parent_dir/results/token_visualisation/ISIC2016/test/mask --intergrated-output $parent_dir/results/token_visualisation/ISIC2016/test/uncertain_vis --image $parent_dir/data/ISIC2016/input/test/ISIC2016_img --points-csv $parent_dir/data/ISIC2016/input/ISIC2016_test_prompts_42.csv --color $color

    # echo "Predicting ISIC2017..."
    # rm -rf $parent_dir/results/token_visualisation/ISIC2017/test/uncertain_vis
    # python uncertain_vis.py --predict-input $parent_dir/results/token_visualisation/ISIC2017/test/mask --intergrated-output $parent_dir/results/token_visualisation/ISIC2017/test/uncertain_vis --image $parent_dir/data/ISIC2017/input/test/ISIC2017_img --points-csv $parent_dir/data/ISIC2017/input/ISIC2017_test_prompts_42.csv --color $color

    # echo "Predicting PH2..."
    # rm -rf $parent_dir/results/token_visualisation/PH2/test/uncertain_vis
    # python uncertain_vis.py --predict-input $parent_dir/results/token_visualisation/PH2/test/mask --intergrated-output $parent_dir/results/token_visualisation/PH2/test/uncertain_vis --image $parent_dir/data/PH2/input/test/PH2_img --points-csv $parent_dir/data/PH2/input/PH2_test_prompts_42.csv --color $color

    # echo "Predicting AtlasZoomIn10..."
    # rm -rf $parent_dir/results/token_visualisation/AtlasZoomIn10/test/uncertain_vis
    # python uncertain_vis.py --predict-input $parent_dir/results/token_visualisation/AtlasZoomIn10/test/mask --intergrated-output $parent_dir/results/token_visualisation/AtlasZoomIn10/test/uncertain_vis --image $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_img --points-csv $parent_dir/data/AtlasZoomIn10/input/AtlasZoomIn10_test_prompts_42.csv --color $color
    
    # echo "Predicting Dermofit..."
    # rm -rf $parent_dir/results/token_visualisation/Dermofit/test/uncertain_vis
    # python uncertain_vis.py --predict-input $parent_dir/results/token_visualisation/Dermofit/test/mask --intergrated-output $parent_dir/results/token_visualisation/Dermofit/test/uncertain_vis --image $parent_dir/data/Dermofit/input/test/Dermofit_img --points-csv $parent_dir/data/Dermofit/input/Dermofit_test_prompts_42.csv --color $color
    
    echo "✅ Evaluate finished."
    echo "#########################################################################"
fi


############################
# add_mask                 #
############################
target_dir=$parent_dir/results/figs/token_visualisation
save_dir=$parent_dir/results/figs/token_visualisation
if [ ! -d "$target_dir" ]; then
    mkdir -p "$target_dir"
fi
dir="HAM10000"
model="PSAM"
dataset="val"
if [ "$add_mask" = true ]; then
    model_dir="$parent_dir/results/$model/$dir/$dataset/mask"
    source_dir="$parent_dir/data/$dir"
    # 复制 Ground Truth 和 Original Image
    img_dir="$source_dir/input/$dataset/${dir}_img"
    gt_dir="$source_dir/input/$dataset/${dir}_seg"
    
    if [ -f "$img_dir/$filename_var" ]; then
        cp "$img_dir/$filename_var" "$target_dir/${dir}_Original_${filename_var}"
    else
        echo "Original image $filename_var does not exist in $img_dir"
    fi

    gt_filename_var="${filename_var%.jpg}_segmentation.png"
    if [ -f "$gt_dir/$gt_filename_var" ]; then
        cp "$gt_dir/$gt_filename_var" "$target_dir/${dir}_GroundTruth_${filename_var}"
    else
        echo "Ground truth $gt_filename_var does not exist in $gt_dir"
    fi


    if [ -f "$model_dir/$filename_var" ]; then
        new_filename="${dir}_${model}_${filename_var}"
        cp "$model_dir/$filename_var" "$target_dir/$new_filename"

        source_path=$target_dir/${dir}_Original_${filename_var}
        gt_mask_path=$target_dir/${dir}_GroundTruth_${filename_var}
        pred_mask_path=$target_dir/$new_filename
        python $parent_dir/code/py_scripts/figs/add_mask.py --source-path $source_path --pred-mask-path $pred_mask_path --gt-mask-path $gt_mask_path --save-dir $save_dir
        echo "✅ Add mask to" $target_dir/$new_filename
    fi
fi