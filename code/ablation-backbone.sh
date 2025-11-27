#!/bin/bash
gpu_sum=2
gpu_id=1,2

RANDOM=$(date +%s| cut -c 6-10)

# Suspending the script if any command fails
set -e
parent_dir=$(dirname "$(pwd)")
current_user=$(whoami)

# Resizing image
split_and_resize=false
img_resize=1024
ham_train_rate=0.8
ham_val_rate=0.1
max_epoch_num=17
# Prompts labeller No.
labeller=42

n_sample_points=4

scribble_type=""

# points or pseudo-box
prompt_type="P_B"

# for vit_type in "vit_h" "vit_l"; do
for vit_type in "vit_l"; do

echo "Current vit_type: $vit_type"

# Output folder name in results folder
output_folder_name="PSAM_cp_"$n_sample_points"_"$prompt_type"_"$vit_type

if [ "$scribble_type" = "" ]; then
    echo "Training model without scribble..."
else
    echo "Training model with scribble:" $scribble_type
    output_folder_name="PSAM_"$scribble_type
fi

check_dataset=false

# Training model
train_model=true
evaluate_model=true
calculate_metrics=true
upload=true

wandb_comment="max_epoch_num:32. Original training. Three specific positive points and four negative points, box+point+noisy mask on specific point, add 4 negative points"


if [ "$scribble_type" = "LinePAndLineN" ]; then
    wandb_comment="Use Line as positive and negative prompt; "$wandb_comment
elif [ "$scribble_type" = "XPAndLineN" ]; then
    wandb_comment="Use X as positive and Line as negative prompt; "$wandb_comment
fi



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

# relogin wandb
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
    wget -c -O "PH2Dataset.rar" "https://example.com/placeholder/download"
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
    wget -c -O "AtlasDataset.zip" "https://example.com/placeholder/download"
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
    wget -c -O "DermofitDataset.zip" "https://example.com/placeholder/download"
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
download_url="https://example.com/placeholder/download"
checkpoint_dir="$(dirname "$checkpoint_file")"
if [ ! -f "$checkpoint_file" ]; then
    echo "$checkpoint_file not found, downloading.."
    wget -O "$checkpoint_file" "$download_url"
fi


checkpoint_file="./pretrained_checkpoint/sam_vit_h_4b8939.pth"
download_url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
checkpoint_dir="$(dirname "$checkpoint_file")"
if [ ! -f "$checkpoint_file" ]; then
    echo "$checkpoint_file not found, downloading.."
    wget -c -O "$checkpoint_file" "$download_url"
fi

checkpoint_file="./pretrained_checkpoint/sam_vit_l_0b3195.pth"
download_url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
checkpoint_dir="$(dirname "$checkpoint_file")"
if [ ! -f "$checkpoint_file" ]; then
    echo "$checkpoint_file not found, downloading.."
    wget -c -O "$checkpoint_file" "$download_url"
fi

echo "✅ Pretrained model check pass."
echo "#########################################################################"

############################
# Train the model         ##
############################
if [ "$train_model" = true ]; then
    echo "Training the model..."
    mkdir -p $parent_dir/results/
    cd $parent_dir/code
    if [ "$vit_type" = "vit_h" ]; then
        echo "pass"
        # CUDA_VISIBLE_DEVICES=$gpu_id torchrun  --master_port=$(($RANDOM % 49152 + 10000 )) --nproc_per_node=$gpu_sum train.py --batch_size_train 4 --checkpoint ./pretrained_checkpoint/sam_vit_h_4b8939.pth --model-type $vit_type --output $parent_dir/results/$output_folder_name --input $parent_dir/data --input_size $img_resize $img_resize --max_epoch_num $max_epoch_num --model_save_fre 2 --logfile $parent_dir/results/$output_folder_name/PSAM_train.txt  --comments "$wandb_comment" --labeller $labeller
    elif [ "$vit_type" = "vit_l" ]; then
        CUDA_VISIBLE_DEVICES=$gpu_id torchrun  --master_port=$(($RANDOM % 49152 + 10000 )) --nproc_per_node=$gpu_sum train.py --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth --model-type $vit_type --output $parent_dir/results/$output_folder_name --input $parent_dir/data --input_size $img_resize $img_resize --max_epoch_num $max_epoch_num --model_save_fre 2 --logfile $parent_dir/results/$output_folder_name/PSAM_train.txt  --comments "$wandb_comment" --labeller $labeller
    else
        echo "Invalid $vit_type: $$vit_type"
        exit 1
    fi
    echo "✅ Training finished."
    echo "#########################################################################"
fi


############################
# Evaluate the model      ##
############################
if [ "$evaluate_model" = true ]; then
    echo "Evaluate the model..."
    cd $parent_dir/code
    # rm -rf $parent_dir/results/$output_folder_name/HAM10000
    # rm -rf $parent_dir/results/$output_folder_name/ISIC2016
    # rm -rf $parent_dir/results/$output_folder_name/ISIC2017
    # rm -rf $parent_dir/results/$output_folder_name/Atlas
    # rm -rf $parent_dir/results/$output_folder_name/Dermofit
    # rm -rf $parent_dir/results/$output_folder_name/PH2


    mkdir -p $parent_dir/results/$output_folder_name
    if [ "$vit_type" = "vit_h" ]; then
        CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nnodes=1 --nproc_per_node=1  --master_port=$(( $RANDOM % 49152 + 10000 )) train.py --eval --checkpoint ./pretrained_checkpoint/sam_vit_h_4b8939.pth --model-type $vit_type --output $parent_dir/results/$output_folder_name --restore-model $parent_dir/results/$output_folder_name/epoch_16.pth --input $parent_dir/data --input_size $img_resize $img_resize --logfile $parent_dir/results/$output_folder_name/PSAM_eva.txt --visualize  --comments "$wandb_comment" --labeller $labeller --n-sample-points $n_sample_points
    elif [ "$vit_type" = "vit_l" ]; then
        CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nnodes=1 --nproc_per_node=1  --master_port=$(( $RANDOM % 49152 + 10000 )) train.py --eval --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth --model-type $vit_type --output $parent_dir/results/$output_folder_name --restore-model $parent_dir/results/$output_folder_name/epoch_16.pth --input $parent_dir/data --input_size $img_resize $img_resize --logfile $parent_dir/results/$output_folder_name/PSAM_eva.txt --visualize  --comments "$wandb_comment" --labeller $labeller --n-sample-points $n_sample_points
    
    else
        echo "Invalid $vit_type: $$vit_type"
        exit 1
    fi
    
    echo "✅ Evaluate finished."
    echo "#########################################################################"
fi

############################
# Calculate metrics       ##
############################
if [ "$calculate_metrics" = true ]; then
    echo "Checking output..."
    cd $parent_dir/code
    python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_val_img --data-path $parent_dir/results/$output_folder_name/HAM10000/val/mask/ --image-size $img_resize $img_resize --image-num 1001
    python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_test_img --data-path $parent_dir/results/$output_folder_name/HAM10000/test/mask/ --image-size $img_resize $img_resize --image-num 1002
    python py_scripts/preprocessing/check_image_size.py --data-name ISIC2016_test_img --data-path $parent_dir/results/$output_folder_name/ISIC2016/test/mask/ --image-size $img_resize $img_resize --image-num 379
    python py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_test_img --data-path $parent_dir/results/$output_folder_name/ISIC2017/test/mask/ --image-size $img_resize $img_resize --image-num 600
    python py_scripts/preprocessing/check_image_size.py --data-name PH2_test_img --data-path $parent_dir/results/$output_folder_name/PH2/test/mask/ --image-size $img_resize $img_resize --image-num 200
    python py_scripts/preprocessing/check_image_size.py --data-name Dermofit_test_img --data-path $parent_dir/results/$output_folder_name/Dermofit/test/mask/ --image-size $img_resize $img_resize --image-num 1300
    # # python py_scripts/preprocessing/check_image_size.py --data-name Atlas_test_img --data-path $parent_dir/results/$output_folder_name/Atlas/test/mask/ --image-size $img_resize $img_resize --image-num 960
    python py_scripts/preprocessing/check_image_size.py --data-name AtlasZoomIn10_test_img --data-path $parent_dir/results/$output_folder_name/AtlasZoomIn10/test/mask/ --image-size $img_resize $img_resize --image-num 960
    # python py_scripts/preprocessing/check_image_size.py --data-name AtlasZoomIn35_test_img --data-path $parent_dir/results/$output_folder_name/AtlasZoomIn35/test/mask/ --image-size $img_resize $img_resize --image-num 960
    echo "[√] Output check pass."

    echo "Calculate PSAM_timecost from csv..."
    cd $parent_dir/code/py_scripts/final
    python calculate_timecost.py --input-csv-path $parent_dir/results/$output_folder_name/HAM10000/val/timecost.csv --output-txt-path $parent_dir/results/$output_folder_name/HAM10000/val/PSAM_timecost.txt --images-num 1001
    python calculate_timecost.py --input-csv-path $parent_dir/results/$output_folder_name/HAM10000/test/timecost.csv --output-txt-path $parent_dir/results/$output_folder_name/HAM10000/test/PSAM_timecost.txt --images-num 1002
    python calculate_timecost.py --input-csv-path $parent_dir/results/$output_folder_name/ISIC2016/test/timecost.csv --output-txt-path $parent_dir/results/$output_folder_name/ISIC2016/test/PSAM_timecost.txt --images-num 379
    python calculate_timecost.py --input-csv-path $parent_dir/results/$output_folder_name/ISIC2017/test/timecost.csv --output-txt-path $parent_dir/results/$output_folder_name/ISIC2017/test/PSAM_timecost.txt --images-num 600
    python calculate_timecost.py --input-csv-path $parent_dir/results/$output_folder_name/PH2/test/timecost.csv --output-txt-path $parent_dir/results/$output_folder_name/PH2/test/PSAM_timecost.txt --images-num 200
    python calculate_timecost.py --input-csv-path $parent_dir/results/$output_folder_name/Dermofit/test/timecost.csv --output-txt-path $parent_dir/results/$output_folder_name/Dermofit/test/PSAM_timecost.txt --images-num 1300
    # # python calculate_timecost.py --input-csv-path $parent_dir/results/$output_folder_name/Atlas/test/timecost.csv --output-txt-path $parent_dir/results/$output_folder_name/Atlas/test/PSAM_timecost.txt --images-num 960
    python calculate_timecost.py --input-csv-path $parent_dir/results/$output_folder_name/AtlasZoomIn10/test/timecost.csv --output-txt-path $parent_dir/results/$output_folder_name/AtlasZoomIn10/test/PSAM_timecost.txt --images-num 960
    # python calculate_timecost.py --input-csv-path $parent_dir/results/$output_folder_name/AtlasZoomIn35/test/timecost.csv --output-txt-path $parent_dir/results/$output_folder_name/AtlasZoomIn35/test/PSAM_timecost.txt --images-num 960
    echo "[√] PSAM_timecost calculation finish."

    cd $parent_dir/code/compare_models/
    python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/HAM10000/input/val/HAM10000_seg --predict-output $parent_dir/results/$output_folder_name/HAM10000/val/mask --csv-output-path $parent_dir/results/$output_folder_name/HAM10000/val/PSAM_metrics.csv --image-size $img_resize $img_resize
    python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/HAM10000/input/test/HAM10000_seg --predict-output $parent_dir/results/$output_folder_name/HAM10000/test/mask --csv-output-path $parent_dir/results/$output_folder_name/HAM10000/test/PSAM_metrics.csv --image-size $img_resize $img_resize
    python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/ISIC2016/input/test/ISIC2016_seg --predict-output $parent_dir/results/$output_folder_name/ISIC2016/test/mask --csv-output-path $parent_dir/results/$output_folder_name/ISIC2016/test/PSAM_metrics.csv --image-size $img_resize $img_resize
    python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/ISIC2017/input/test/ISIC2017_seg --predict-output $parent_dir/results/$output_folder_name/ISIC2017/test/mask --csv-output-path $parent_dir/results/$output_folder_name/ISIC2017/test/PSAM_metrics.csv --image-size $img_resize $img_resize
    python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/PH2/input/test/PH2_seg --predict-output $parent_dir/results/$output_folder_name/PH2/test/mask --csv-output-path $parent_dir/results/$output_folder_name/PH2/test/PSAM_metrics.csv --image-size $img_resize $img_resize
    python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/Dermofit/input/test/Dermofit_seg --predict-output $parent_dir/results/$output_folder_name/Dermofit/test/mask --csv-output-path $parent_dir/results/$output_folder_name/Dermofit/test/PSAM_metrics.csv --image-size $img_resize $img_resize
    # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/Atlas/input/test/Atlas_seg --predict-output $parent_dir/results/$output_folder_name/Atlas/test/mask --csv-output-path $parent_dir/results/$output_folder_name/Atlas/test/PSAM_metrics.csv --image-size $img_resize $img_resize
    python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_seg --predict-output $parent_dir/results/$output_folder_name/AtlasZoomIn10/test/mask --csv-output-path $parent_dir/results/$output_folder_name/AtlasZoomIn10/test/PSAM_metrics.csv --image-size $img_resize $img_resize
    # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/AtlasZoomIn35/input/test/AtlasZoomIn35_seg --predict-output $parent_dir/results/$output_folder_name/AtlasZoomIn35/test/mask --csv-output-path $parent_dir/results/$output_folder_name/AtlasZoomIn35/test/PSAM_metrics.csv --image-size $img_resize $img_resize

    echo "✅ Calculate metrics finished."
    echo "#########################################################################"
fi
############################
# Upload wandb            ##
############################
if [ "$upload" = true ]; then
    cd $parent_dir/code/compare_models/
    python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name Abl_PSAM_$n_sample_points$prompt_type$vit_type-HAM10000_val --metrics-csv $parent_dir/results/$output_folder_name/HAM10000/val/PSAM_metrics.csv --timecost-txt $parent_dir/results/$output_folder_name/HAM10000/val/PSAM_timecost.txt --images-num 1001 --comments "$wandb_comment" --labeller $labeller
    python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name Abl_PSAM_$n_sample_points$prompt_type$vit_type-HAM10000_test --metrics-csv $parent_dir/results/$output_folder_name/HAM10000/test/PSAM_metrics.csv --timecost-txt $parent_dir/results/$output_folder_name/HAM10000/test/PSAM_timecost.txt --images-num 1002 --comments "$wandb_comment" --labeller $labeller
    python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name Abl_PSAM_$n_sample_points$prompt_type$vit_type-ISIC2016_test --metrics-csv $parent_dir/results/$output_folder_name/ISIC2016/test/PSAM_metrics.csv --timecost-txt $parent_dir/results/$output_folder_name/ISIC2016/test/PSAM_timecost.txt --images-num 379 --comments "$wandb_comment" --labeller $labeller
    python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name Abl_PSAM_$n_sample_points$prompt_type$vit_type-ISIC2017_test --metrics-csv $parent_dir/results/$output_folder_name/ISIC2017/test/PSAM_metrics.csv --timecost-txt $parent_dir/results/$output_folder_name/ISIC2017/test/PSAM_timecost.txt --images-num 600 --comments "$wandb_comment" --labeller $labeller
    python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name Abl_PSAM_$n_sample_points$prompt_type$vit_type-PH2_test --metrics-csv $parent_dir/results/$output_folder_name/PH2/test/PSAM_metrics.csv --timecost-txt $parent_dir/results/$output_folder_name/PH2/test/PSAM_timecost.txt --images-num 200 --comments "$wandb_comment" --labeller $labeller
    python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name Abl_PSAM_$n_sample_points$prompt_type$vit_type-Dermofit_test --metrics-csv $parent_dir/results/$output_folder_name/Dermofit/test/PSAM_metrics.csv --timecost-txt $parent_dir/results/$output_folder_name/Dermofit/test/PSAM_timecost.txt --images-num 1300 --comments "$wandb_comment" --labeller $labeller
    # # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name PSAM-Atlas_test --metrics-csv $parent_dir/results/$output_folder_name/Atlas/test/PSAM_metrics.csv --timecost-txt $parent_dir/results/$output_folder_name/Atlas/test/PSAM_timecost.txt --images-num 960 --comments "$wandb_comment" --labeller $labeller
    python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name Abl_PSAM_$n_sample_points$prompt_type-AtlasZoomIn10_test --metrics-csv $parent_dir/results/$output_folder_name/AtlasZoomIn10/test/PSAM_metrics.csv --timecost-txt $parent_dir/results/$output_folder_name/AtlasZoomIn10/test/PSAM_timecost.txt --images-num 960 --comments "$wandb_comment" --labeller $labeller
    # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name PSAM-AtlasZoomIn35_test --metrics-csv $parent_dir/results/$output_folder_name/AtlasZoomIn35/test/PSAM_metrics.csv --timecost-txt $parent_dir/results/$output_folder_name/AtlasZoomIn35/test/PSAM_timecost.txt --images-num 960 --comments "$wandb_comment" --labeller $labeller
    echo "✅ PSAM Metrics upload finished."
    echo "#########################################################################"
fi
done
