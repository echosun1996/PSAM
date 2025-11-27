#!/bin/bash
gpu_id=1
img_resize=1024
generate_prompts=false
covert_binary_mask=false
labeller=42


# [Finish] Unet Settings
COMPARE_UNET=false
retrain_UNet=false
evaluate_UNet=false
calculate_metrics_UNet=false
upload_UNet=false

# [Finish] nnUnet Settings
COMPARE_NNUNET=false
retrain_nnUNet=false
evaluate_nnUNet=false
calculate_metrics_nnUNet=false
upload_nnUNet=false

# [Finish] SAM Settings (auto segmention)
COMPARE_SAMAUTO=false
evaluate_SAMAuto=false
calculate_metrics_SAMAuto=false
upload_SAMAuto=false

# SAM Settings (point prompts)
COMPARE_SAMSPECIFIC=false
evaluate_SAMSpecific=false
calculate_metrics_SAMSpecific=false
upload_SAMSpecific=false

# [Finish] One-Prompt Settings
COMPARE_ONEPROMPT=false
retrain_OnePrompt=false
evaluate_OnePrompt=false
calculate_metrics_OnePrompt=false
upload_OnePrompt=false


# [TODO] MedSAM Settings
COMPARE_MedSAM=false
retrain_MedSAM=false
evaluate_MedSAM=true
calculate_metrics_MedSAM=true
upload_MedSAM=true


# [BOX] MedSAMAuto Settings
COMPARE_MedSAMAuto=true
retrain_MedSAMAuto=false
evaluate_MedSAMAuto=false
calculate_metrics_MedSAMAuto=false
upload_MedSAMAuto=true

# Scribble Saliency (CVPR2020)
COMPARE_SCRIBBLESALIENCY=false
generate_dataset_ScribbleSaliency=false
retrain_ScribbleSaliency=false
evaluate_ScribbleSaliency=false
calculate_metrics_ScribbleSaliency=false
upload_ScribbleSaliency=false

# ScribblePrompt Settings
COMPARE_SCRIBBLEPROMPT=false
# retrain_ScribblePrompt=true
# evaluate_ScribblePrompt=false
# calculate_metrics_ScribblePrompt=false
# upload_ScribblePrompt=true

# [Remove] WSCOD Settings
# COMPARE_WSCOD=false
# scribbles_generator_WSCOD=false
# prepare_WSCOD=false
# retrain_WSCOD=true
# evaluate_WSCOD=false
# calculate_metrics_WSCOD=false
# upload_WSCOD=true

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

# relogin wandb
# relogin wandb (use environment variable WANDB_API_KEY if set)
if [ -n "$WANDB_API_KEY" ]; then
    wandb login --relogin "$WANDB_API_KEY"
else
    echo "Warning: WANDB_API_KEY not set. Skipping wandb login."
    echo "To use wandb, set WANDB_API_KEY environment variable."
fi
wandb sync --clean

# Check data folder
if [ ! -d "$parent_dir/data" ]; then
    echo "Error: 'data' folder not found in the parent directory."
    exit 1
fi

############################
# Check resize dataset    ##
############################
# echo "Checking image size..."
# cd $parent_dir/code
# python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_train_img --data-path $parent_dir/data/HAM10000/input/train/HAM10000_img/ --image-size $img_resize $img_resize --image-num 8012
# python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_train_seg --data-path $parent_dir/data/HAM10000/input/train/HAM10000_seg/ --image-size $img_resize $img_resize --image-num 8012

# python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_val_img --data-path $parent_dir/data/HAM10000/input/val/HAM10000_img/ --image-size $img_resize $img_resize --image-num 1001
# python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_val_seg --data-path $parent_dir/data/HAM10000/input/val/HAM10000_seg/ --image-size $img_resize $img_resize --image-num 1001

# python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_test_img --data-path $parent_dir/data/HAM10000/input/test/HAM10000_img/ --image-size $img_resize $img_resize --image-num 1002
# python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_test_seg --data-path $parent_dir/data/HAM10000/input/test/HAM10000_seg/ --image-size $img_resize $img_resize --image-num 1002

# python py_scripts/preprocessing/check_image_size.py --data-name ISIC2016_train_img --data-path $parent_dir/data/ISIC2016/input/train/ISIC2016_img/ --image-size $img_resize $img_resize --image-num 900
# python py_scripts/preprocessing/check_image_size.py --data-name ISIC2016_train_seg --data-path $parent_dir/data/ISIC2016/input/train/ISIC2016_seg/ --image-size $img_resize $img_resize --image-num 900

# python py_scripts/preprocessing/check_image_size.py --data-name ISIC2016_test_img --data-path $parent_dir/data/ISIC2016/input/test/ISIC2016_img/ --image-size $img_resize $img_resize --image-num 379
# python py_scripts/preprocessing/check_image_size.py --data-name ISIC2016_test_seg --data-path $parent_dir/data/ISIC2016/input/test/ISIC2016_seg/ --image-size $img_resize $img_resize --image-num 379

# python py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_train_img --data-path $parent_dir/data/ISIC2017/input/train/ISIC2017_img/ --image-size $img_resize $img_resize --image-num 2000
# python py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_train_seg --data-path $parent_dir/data/ISIC2017/input/train/ISIC2017_seg/ --image-size $img_resize $img_resize --image-num 2000

# python py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_val_img --data-path $parent_dir/data/ISIC2017/input/val/ISIC2017_img/ --image-size $img_resize $img_resize --image-num 150
# python py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_val_seg --data-path $parent_dir/data/ISIC2017/input/val/ISIC2017_seg/ --image-size $img_resize $img_resize --image-num 150

# python py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_test_img --data-path $parent_dir/data/ISIC2017/input/test/ISIC2017_img/ --image-size $img_resize $img_resize --image-num 600
# python py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_test_seg --data-path $parent_dir/data/ISIC2017/input/test/ISIC2017_seg/ --image-size $img_resize $img_resize --image-num 600

# python py_scripts/preprocessing/check_image_size.py --data-name PH2_test_img --data-path $parent_dir/data/PH2/input/test/PH2_img/ --image-size $img_resize $img_resize --image-num 200
# python py_scripts/preprocessing/check_image_size.py --data-name PH2_test_seg --data-path $parent_dir/data/PH2/input/test/PH2_seg/ --image-size $img_resize $img_resize --image-num 200

# python py_scripts/preprocessing/check_image_size.py --data-name Atlas_test_img --data-path $parent_dir/data/Atlas/input/test/Atlas_img/ --image-size $img_resize $img_resize --image-num 960
# python py_scripts/preprocessing/check_image_size.py --data-name Atlas_test_seg --data-path $parent_dir/data/Atlas/input/test/Atlas_seg/ --image-size $img_resize $img_resize --image-num 960
# echo "✅ Dataset check pass."
# echo "#########################################################################"


if [ "$covert_binary_mask" = true ]; then
    cd $parent_dir/code/py_scripts/preprocessing/
    python mask_to_binary.py --input $parent_dir/data/HAM10000/input/train/HAM10000_seg/ --output $parent_dir/data/HAM10000/input/train/HAM10000_seg_binary/
    python mask_to_binary.py --input $parent_dir/data/HAM10000/input/val/HAM10000_seg/ --output $parent_dir/data/HAM10000/input/val/HAM10000_seg_binary/
    python mask_to_binary.py --input $parent_dir/data/HAM10000/input/test/HAM10000_seg/ --output $parent_dir/data/HAM10000/input/test/HAM10000_seg_binary/
fi


if [ "$generate_prompts" = true ]; then
    cd $parent_dir/code/py_scripts/preprocessing/
    # Test for vis
    # python generate_prompts.py --img-path $parent_dir/data/HAM10000/input/train/HAM10000_img/ --mask-path $parent_dir/data/HAM10000/input/train/HAM10000_seg/ --prompt-output $parent_dir/data/HAM10000/input/train/HAM10000_seg/HAM10000_prompts.csv --vis-path ./temp
    
    # python generate_prompts.py --random-seed $labeller --mask-path $parent_dir/data/HAM10000/input/train/HAM10000_seg/ --prompt-output $parent_dir/data/HAM10000/input/HAM10000_train_prompts_$labeller.csv
    # python generate_prompts.py --random-seed $labeller --mask-path $parent_dir/data/HAM10000/input/val/HAM10000_seg/ --prompt-output $parent_dir/data/HAM10000/input/HAM10000_val_prompts_$labeller.csv
    # python generate_prompts.py --random-seed $labeller --mask-path $parent_dir/data/HAM10000/input/test/HAM10000_seg/ --prompt-output $parent_dir/data/HAM10000/input/HAM10000_test_prompts_$labeller.csv

    # python generate_prompts.py --random-seed $labeller --mask-path $parent_dir/data/ISIC2016/input/train/ISIC2016_seg/ --prompt-output $parent_dir/data/ISIC2016/input/ISIC2016_train_prompts_$labeller.csv
    # python generate_prompts.py --random-seed $labeller --mask-path $parent_dir/data/ISIC2016/input/test/ISIC2016_seg/ --prompt-output $parent_dir/data/ISIC2016/input/ISIC2016_test_prompts_$labeller.csv

    # python generate_prompts.py --random-seed $labeller --mask-path $parent_dir/data/ISIC2017/input/train/ISIC2017_seg/ --prompt-output $parent_dir/data/ISIC2017/input/ISIC2017_train_prompts_$labeller.csv
    # python generate_prompts.py --random-seed $labeller --mask-path $parent_dir/data/ISIC2017/input/val/ISIC2017_seg/ --prompt-output $parent_dir/data/ISIC2017/input/ISIC2017_val_prompts_$labeller.csv
    # python generate_prompts.py --random-seed $labeller --mask-path $parent_dir/data/ISIC2017/input/test/ISIC2017_seg/ --prompt-output $parent_dir/data/ISIC2017/input/ISIC2017_test_prompts_$labeller.csv

    # python generate_prompts.py --random-seed $labeller --mask-path $parent_dir/data/PH2/input/test/PH2_seg/ --prompt-output $parent_dir/data/PH2/input/PH2_test_prompts_$labeller.csv

    # python generate_prompts.py --random-seed $labeller --mask-path $parent_dir/data/Atlas/input/test/Atlas_seg/ --prompt-output $parent_dir/data/Atlas/input/Atlas_test_prompts_$labeller.csv
    # python generate_prompts.py --random-seed $labeller --mask-path $parent_dir/data/AtlasZoomIn/input/test/AtlasZoomIn_seg/ --prompt-output $parent_dir/data/AtlasZoomIn/input/AtlasZoomIn_test_prompts_$labeller.csv
    python generate_prompts.py --random-seed $labeller --mask-path $parent_dir/data/AtlasZoomIn35/input/test/AtlasZoomIn35_seg/ --prompt-output $parent_dir/data/AtlasZoomIn35/input/AtlasZoomIn35_test_prompts_$labeller.csv

    # python generate_prompts.py --random-seed $labeller --mask-path $parent_dir/data/Dermofit/input/test/Dermofit_seg/ --prompt-output $parent_dir/data/Dermofit/input/Dermofit_test_prompts_$labeller.csv
fi

# ############################
# #  UNet                   ##
# ############################
if [ "$COMPARE_UNET" = true ]; then
    echo "Comparing with UNet..."
    mkdir -p $parent_dir/code/compare_models/reps/
    cd $parent_dir/code/compare_models/reps/
    if [ -d "UNet" ];
    then
        echo "[√] UNet already exists."
    else
        git clone https://github.com/milesial/Pytorch-UNet.git UNet
    fi

    if [ "$retrain_UNet" = true ]; then
        python $parent_dir/code/compare_models/UNet_train.py --scale 1 --train-img-path $parent_dir/data/HAM10000/input/train/HAM10000_img/ --train-seg-path $parent_dir/data/HAM10000/input/train/HAM10000_seg/ --val-img-path $parent_dir/data/HAM10000/input/val/HAM10000_img/ --val-seg-path $parent_dir/data/HAM10000/input/val/HAM10000_seg/ --checkpoints $parent_dir/code/compare_models/checkpoints/UNet/
        echo "[√] UNet Training finished."
    fi

    if [ "$evaluate_UNet" = true ]; then
        cd $parent_dir/code/compare_models/

        # # HAM10000 val
        # start_time=$(date +%s)
        # python UNet_predict.py --input $parent_dir/data/HAM10000/input/val/HAM10000_img/ --output $parent_dir/data/HAM10000/output/val/UNet
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/HAM10000/output/val/UNet_timecost.txt
        # python $parent_dir/code/py_scripts/final/binary2visable.py --path $parent_dir/data/HAM10000/output/val/UNet
        # echo "[√] HAM10000 val finished."

        # # HAM10000 test
        # start_time=$(date +%s)
        # python UNet_predict.py --input $parent_dir/data/HAM10000/input/test/HAM10000_img/ --output $parent_dir/data/HAM10000/output/test/UNet
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/HAM10000/output/test/UNet_timecost.txt
        # python $parent_dir/code/py_scripts/final/binary2visable.py --path $parent_dir/data/HAM10000/output/test/UNet
        # echo "[√] HAM10000 test finished."

        # # ISIC2016 test
        # start_time=$(date +%s)
        # python UNet_predict.py --input $parent_dir/data/ISIC2016/input/test/ISIC2016_img/ --output $parent_dir/data/ISIC2016/output/test/UNet
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/ISIC2016/output/test/UNet_timecost.txt
        # python $parent_dir/code/py_scripts/final/binary2visable.py --path $parent_dir/data/ISIC2016/output/test/UNet
        # echo "[√] ISIC2016 val finished."
        
        # # ISIC2017
        # start_time=$(date +%s)
        # python UNet_predict.py --input $parent_dir/data/ISIC2017/input/test/ISIC2017_img/ --output $parent_dir/data/ISIC2017/output/test/UNet
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/ISIC2017/output/test/UNet_timecost.txt
        # python $parent_dir/code/py_scripts/final/binary2visable.py --path $parent_dir/data/ISIC2017/output/test/UNet
        # echo "[√] ISIC2017 val finished."
        
        # # PH2
        # start_time=$(date +%s)
        # python UNet_predict.py --input $parent_dir/data/PH2/input/test/PH2_img/ --output $parent_dir/data/PH2/output/test/UNet
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/PH2/output/test/UNet_timecost.txt
        # python $parent_dir/code/py_scripts/final/binary2visable.py --path $parent_dir/data/PH2/output/test/UNet
        # echo "[√] PH2 val finished."

        # # Dermofit
        # start_time=$(date +%s)
        # python UNet_predict.py --input $parent_dir/data/Dermofit/input/test/Dermofit_img/ --output $parent_dir/data/Dermofit/output/test/UNet
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/Dermofit/output/test/UNet_timecost.txt
        # python $parent_dir/code/py_scripts/final/binary2visable.py --path $parent_dir/data/Dermofit/output/test/UNet
        # echo "[√] Dermofit val finished."
        
        # # Atlas
        # start_time=$(date +%s)
        # python UNet_predict.py --input $parent_dir/data/Atlas/input/test/Atlas_img/ --output $parent_dir/data/Atlas/output/test/UNet
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/Atlas/output/test/UNet_timecost.txt
        # python $parent_dir/code/py_scripts/final/binary2visable.py --path $parent_dir/data/Atlas/output/test/UNet
        # echo "[√] Atlas val finished."
        
        # Atlas ZoomIn 10
        start_time=$(date +%s)
        python UNet_predict.py --input $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_img/ --output $parent_dir/data/AtlasZoomIn10/output/test/UNet
        end_time=$(date +%s)
        cost_time=$((end_time - start_time))
        echo "Total cost [$cost_time] seconds" > $parent_dir/data/AtlasZoomIn10/output/test/UNet_timecost.txt
        python $parent_dir/code/py_scripts/final/binary2visable.py --path $parent_dir/data/AtlasZoomIn10/output/test/UNet
        echo "[√] Atlas Zoom 10 val finished."

        # Atlas ZoomIn 35
        start_time=$(date +%s)
        python UNet_predict.py --input $parent_dir/data/AtlasZoomIn35/input/test/AtlasZoomIn35_img/ --output $parent_dir/data/AtlasZoomIn35/output/test/UNet
        end_time=$(date +%s)
        cost_time=$((end_time - start_time))
        echo "Total cost [$cost_time] seconds" > $parent_dir/data/AtlasZoomIn35/output/test/UNet_timecost.txt
        python $parent_dir/code/py_scripts/final/binary2visable.py --path $parent_dir/data/AtlasZoomIn35/output/test/UNet
        echo "[√] Atlas Zoom 35 val finished."
    
        echo "[√] UNet Evaluate finished."
    fi

    if [ "$calculate_metrics_UNet" = true ]; then
        cd $parent_dir/code/compare_models/
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/HAM10000/input/val/HAM10000_seg --predict-output $parent_dir/data/HAM10000/output/val/UNet --csv-output-path $parent_dir/data/HAM10000/output/val/UNet_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/HAM10000/input/test/HAM10000_seg --predict-output $parent_dir/data/HAM10000/output/test/UNet --csv-output-path $parent_dir/data/HAM10000/output/test/UNet_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/ISIC2016/input/test/ISIC2016_seg --predict-output $parent_dir/data/ISIC2016/output/test/UNet --csv-output-path $parent_dir/data/ISIC2016/output/test/UNet_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/ISIC2017/input/test/ISIC2017_seg --predict-output $parent_dir/data/ISIC2017/output/test/UNet --csv-output-path $parent_dir/data/ISIC2017/output/test/UNet_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/PH2/input/test/PH2_seg --predict-output $parent_dir/data/PH2/output/test/UNet --csv-output-path $parent_dir/data/PH2/output/test/UNet_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/Dermofit/input/test/Dermofit_seg --predict-output $parent_dir/data/Dermofit/output/test/UNet --csv-output-path $parent_dir/data/Dermofit/output/test/UNet_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/Atlas/input/test/Atlas_seg --predict-output $parent_dir/data/Atlas/output/test/UNet --csv-output-path $parent_dir/data/Atlas/output/test/UNet_metrics.csv --image-size $img_resize $img_resize
        python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_seg --predict-output $parent_dir/data/AtlasZoomIn10/output/test/UNet --csv-output-path $parent_dir/data/AtlasZoomIn10/output/test/UNet_metrics.csv --image-size $img_resize $img_resize
        python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/AtlasZoomIn35/input/test/AtlasZoomIn35_seg --predict-output $parent_dir/data/AtlasZoomIn35/output/test/UNet --csv-output-path $parent_dir/data/AtlasZoomIn35/output/test/UNet_metrics.csv --image-size $img_resize $img_resize
    fi

    if [ "$upload_UNet" = true ]; then
        echo "Checking UNet output..."
        cd $parent_dir/code
        # python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_val_img --data-path $parent_dir/data/HAM10000/output/val/UNet/ --image-size $img_resize $img_resize --image-num 1001
        # python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_test_img --data-path $parent_dir/data/HAM10000/output/test/UNet/ --image-size $img_resize $img_resize --image-num 1002
        # python py_scripts/preprocessing/check_image_size.py --data-name ISIC2016_test_img --data-path $parent_dir/data/ISIC2016/output/test/UNet/ --image-size $img_resize $img_resize --image-num 379
        # python py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_test_img --data-path $parent_dir/data/ISIC2017/output/test/UNet/ --image-size $img_resize $img_resize --image-num 600
        # python py_scripts/preprocessing/check_image_size.py --data-name PH2_test_img --data-path $parent_dir/data/PH2/output/test/UNet/ --image-size $img_resize $img_resize --image-num 200
        # python py_scripts/preprocessing/check_image_size.py --data-name Dermofit_test_img --data-path $parent_dir/data/Dermofit/output/test/UNet/ --image-size $img_resize $img_resize --image-num 1300
        # python py_scripts/preprocessing/check_image_size.py --data-name Atlas_test_img --data-path $parent_dir/data/Atlas/output/test/UNet/ --image-size $img_resize $img_resize --image-num 960
        echo "[√] UNet output check pass."


        cd $parent_dir/code/compare_models/
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name UNet-HAM10000_val --metrics-csv $parent_dir/data/HAM10000/output/val/UNet_metrics.csv --timecost-txt $parent_dir/data/HAM10000/output/val/UNet_timecost.txt --images-num 1001
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name UNet-HAM10000_test --metrics-csv $parent_dir/data/HAM10000/output/test/UNet_metrics.csv --timecost-txt $parent_dir/data/HAM10000/output/test/UNet_timecost.txt --images-num 1002
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name UNet-ISIC2016_test --metrics-csv $parent_dir/data/ISIC2016/output/test/UNet_metrics.csv --timecost-txt $parent_dir/data/ISIC2016/output/test/UNet_timecost.txt --images-num 379
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name UNet-ISIC2017_test --metrics-csv $parent_dir/data/ISIC2017/output/test/UNet_metrics.csv --timecost-txt $parent_dir/data/ISIC2017/output/test/UNet_timecost.txt --images-num 600
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name UNet-PH2_test --metrics-csv $parent_dir/data/PH2/output/test/UNet_metrics.csv --timecost-txt $parent_dir/data/PH2/output/test/UNet_timecost.txt --images-num 200
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name UNet-Dermofit_test --metrics-csv $parent_dir/data/Dermofit/output/test/UNet_metrics.csv --timecost-txt $parent_dir/data/Dermofit/output/test/UNet_timecost.txt --images-num 1300
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name UNet-Atlas_test --metrics-csv $parent_dir/data/Atlas/output/test/UNet_metrics.csv --timecost-txt $parent_dir/data/Atlas/output/test/UNet_timecost.txt --images-num 960
        python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name UNet-AtlasZoomIn10_test --metrics-csv $parent_dir/data/AtlasZoomIn10/output/test/UNet_metrics.csv --timecost-txt $parent_dir/data/AtlasZoomIn10/output/test/UNet_timecost.txt --images-num 960
        python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name UNet-AtlasZoomIn35_test --metrics-csv $parent_dir/data/AtlasZoomIn35/output/test/UNet_metrics.csv --timecost-txt $parent_dir/data/AtlasZoomIn35/output/test/UNet_timecost.txt --images-num 960
        echo "[√] UNet Metrics upload finished."

    fi

    cd $parent_dir/code
    echo "✅ UNet compare finished."
    echo "#########################################################################"
fi

############################
#  nnUNet                 ##
############################
if [ "$COMPARE_NNUNET" = true ]; then
    echo "Comparing with nnUNet..."
    mkdir -p $parent_dir/code/compare_models/reps/
    cd $parent_dir/code/compare_models/reps/
    if [ -d "nnUNet" ];
    then
        echo "[√] nnUNet already exists."
    else
        git clone https://github.com/MIC-DKFZ/nnUNet.git nnUNet
        cd $parent_dir/code/compare_models/reps/nnUNet
        pip install -e .
    fi

    if [ "$retrain_nnUNet" = true ]; then
        cd $parent_dir/code/compare_models/
        # Move to nnUNet data folder
        python nnUNet-train_move.py --input $parent_dir/data/HAM10000/input/ --data-path $parent_dir/data/
        
        # Prrprocess the nnUNet data
        python nnUNet-train_preprocess.py --data-path $parent_dir/data/

        export nnUNet_preprocessed="$parent_dir/data/nnUNetFrame/DATASET/nnUNet_preprocessed"
        export nnUNet_raw="$parent_dir/data/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data"
        export nnUNet_results="$parent_dir/data/nnUNetFrame/DATASET/nnUNet_trained_models"

        # Generate nnUNet training plan
        nnUNetv2_plan_and_preprocess -d 666 --verify_dataset_integrity

        # got training images number
        directory="$parent_dir/data/HAM10000/input/train/HAM10000_img"
        training_image_count=$(find "$directory" -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) | wc -l)
        echo "Training images number: $training_image_count"

        directory="$parent_dir/data/HAM10000/input/val/HAM10000_img"
        val_image_count=$(find "$directory" -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) | wc -l)
        echo "Training images number: $val_image_count"

        # Adjust the split of the dataset
        python nnUNet-train_manual_split.py --json-path $parent_dir/data/nnUNetFrame/DATASET/nnUNet_preprocessed/Dataset666_HAM/splits_final.json --training-image-count $training_image_count --val-image-count $val_image_count
        
        # Train nnUNet 
        CUDA_VISIBLE_DEVICES=$gpu_id nnUNetv2_train 666 2d 0 --c # self.num_epochs = 30 /mnt/hdd/sdb/jiajun/zu52/PSAM/code/compare_models/reps/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py

        echo "[√] nnUNet Training finished."
    fi


    if [ "$evaluate_nnUNet" = true ]; then
        cd $parent_dir/code/compare_models/
        export nnUNet_preprocessed="$parent_dir/data/nnUNetFrame/DATASET/nnUNet_preprocessed"
        export nnUNet_raw="$parent_dir/data/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data"
        export nnUNet_results="$parent_dir/data/nnUNetFrame/DATASET/nnUNet_trained_models"

        # # HAM10000 val
        # echo "Preprocess HAM10000 val..."
        # rm -rf $parent_dir/data/nnUNetFrame/HAM10000_val
        # python nnUNet-predict_move.py --input-img-path $parent_dir/data/HAM10000/input/val/HAM10000_img --input-seg-path $parent_dir/data/HAM10000/input/val/HAM10000_seg --output-path $parent_dir/data/nnUNetFrame/HAM10000_val
        # python nnUNet-predict_preprocess.py --preprocess-path $parent_dir/data/nnUNetFrame/HAM10000_val

        # # HAM10000 Test
        # echo "Preprocess HAM10000 test..."
        # rm -rf $parent_dir/data/nnUNetFrame/HAM10000_test
        # python nnUNet-predict_move.py --input-img-path $parent_dir/data/HAM10000/input/test/HAM10000_img --input-seg-path $parent_dir/data/HAM10000/input/test/HAM10000_seg --output-path $parent_dir/data/nnUNetFrame/HAM10000_test
        # python nnUNet-predict_preprocess.py --preprocess-path $parent_dir/data/nnUNetFrame/HAM10000_test
        # # ISIC2016 Test
        # echo "Preprocess ISIC2016 test..."
        # rm -rf $parent_dir/data/nnUNetFrame/ISIC2016_test
        # python nnUNet-predict_move.py --input-img-path $parent_dir/data/ISIC2016/input/test/ISIC2016_img --input-seg-path $parent_dir/data/ISIC2016/input/test/ISIC2016_seg --output-path $parent_dir/data/nnUNetFrame/ISIC2016_test
        # python nnUNet-predict_preprocess.py --preprocess-path $parent_dir/data/nnUNetFrame/ISIC2016_test
        # # ISIC2017 Test
        # echo "Preprocess ISIC2017 test..."
        # rm -rf $parent_dir/data/nnUNetFrame/ISIC2017_test
        # python nnUNet-predict_move.py --input-img-path $parent_dir/data/ISIC2017/input/test/ISIC2017_img --input-seg-path $parent_dir/data/ISIC2017/input/test/ISIC2017_seg --output-path $parent_dir/data/nnUNetFrame/ISIC2017_test
        # python nnUNet-predict_preprocess.py --preprocess-path $parent_dir/data/nnUNetFrame/ISIC2017_test
        # # PH2
        # echo "Preprocess PH2 test..."
        # rm -rf $parent_dir/data/nnUNetFrame/PH2_test
        # python nnUNet-predict_move.py --input-img-path $parent_dir/data/PH2/input/test/PH2_img --input-seg-path $parent_dir/data/PH2/input/test/PH2_seg --output-path $parent_dir/data/nnUNetFrame/PH2_test
        # python nnUNet-predict_preprocess.py --preprocess-path $parent_dir/data/nnUNetFrame/PH2_test
        # # Dermofit
        # echo "Preprocess Dermofit test..."
        # rm -rf $parent_dir/data/nnUNetFrame/Dermofit_test
        # python nnUNet-predict_move.py --input-img-path $parent_dir/data/Dermofit/input/test/Dermofit_img --input-seg-path $parent_dir/data/Dermofit/input/test/Dermofit_seg --output-path $parent_dir/data/nnUNetFrame/Dermofit_test
        # python nnUNet-predict_preprocess.py --preprocess-path $parent_dir/data/nnUNetFrame/Dermofit_test
        # echo "[√] nnUNet Preprocess finished."
        # # Atlas
        # echo "Preprocess Atlas test..."
        # rm -rf $parent_dir/data/nnUNetFrame/Atlas_test
        # python nnUNet-predict_move.py --input-img-path $parent_dir/data/Atlas/input/test/Atlas_img --input-seg-path $parent_dir/data/Atlas/input/test/Atlas_seg --output-path $parent_dir/data/nnUNetFrame/Atlas_test
        # python nnUNet-predict_preprocess.py --preprocess-path $parent_dir/data/nnUNetFrame/Atlas_test

        # Atlas ZoomIn 10
        echo "Preprocess Atlas test..."
        rm -rf $parent_dir/data/nnUNetFrame/AtlasZoomIn10_test
        python nnUNet-predict_move.py --input-img-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_img --input-seg-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_seg --output-path $parent_dir/data/nnUNetFrame/AtlasZoomIn10_test
        python nnUNet-predict_preprocess.py --preprocess-path $parent_dir/data/nnUNetFrame/AtlasZoomIn10_test

        # Atlas ZoomIn 35
        echo "Preprocess Atlas test..."
        rm -rf $parent_dir/data/nnUNetFrame/AtlasZoomIn35_test
        python nnUNet-predict_move.py --input-img-path $parent_dir/data/AtlasZoomIn35/input/test/AtlasZoomIn35_img --input-seg-path $parent_dir/data/AtlasZoomIn35/input/test/AtlasZoomIn35_seg --output-path $parent_dir/data/nnUNetFrame/AtlasZoomIn35_test
        python nnUNet-predict_preprocess.py --preprocess-path $parent_dir/data/nnUNetFrame/AtlasZoomIn35_test


        echo "[√] nnUNet Preprocess finished."

        # echo "Evaluating on HAM10000 val..."
        # rm -rf $parent_dir/data/HAM10000/output/val/nnUNet
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id nnUNetv2_predict -i $parent_dir/data/nnUNetFrame/HAM10000_val/PredictDataset/imagesTr -o $parent_dir/data/HAM10000/output/val/nnUNet -d 666 -c 2d -f 0
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/HAM10000/output/val/nnUNet_timecost.txt
        # python $parent_dir/code/py_scripts/final/binary2visable.py --path $parent_dir/data/HAM10000/output/val/nnUNet

        # echo "Evaluating on HAM10000 test..."
        # rm -rf $parent_dir/data/HAM10000/output/test/nnUNet
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id nnUNetv2_predict -i $parent_dir/data/nnUNetFrame/HAM10000_test/PredictDataset/imagesTr -o $parent_dir/data/HAM10000/output/test/nnUNet -d 666 -c 2d -f 0
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/HAM10000/output/test/nnUNet_timecost.txt
        # python $parent_dir/code/py_scripts/final/binary2visable.py --path $parent_dir/data/HAM10000/output/test/nnUNet

        # echo "Evaluating on ISIC2016 test..."
        # rm -rf $parent_dir/data/ISIC2016/output/test/nnUNet
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id nnUNetv2_predict -i $parent_dir/data/nnUNetFrame/ISIC2016_test/PredictDataset/imagesTr -o $parent_dir/data/ISIC2016/output/test/nnUNet -d 666 -c 2d -f 0
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/ISIC2016/output/test/nnUNet_timecost.txt
        # python $parent_dir/code/py_scripts/final/binary2visable.py --path $parent_dir/data/ISIC2016/output/test/nnUNet

        # echo "Evaluating on ISIC2017 test..."
        # rm -rf $parent_dir/data/ISIC2017/output/test/nnUNet
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id nnUNetv2_predict -i $parent_dir/data/nnUNetFrame/ISIC2017_test/PredictDataset/imagesTr -o $parent_dir/data/ISIC2017/output/test/nnUNet -d 666 -c 2d -f 0
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/ISIC2017/output/test/nnUNet_timecost.txt
        # python $parent_dir/code/py_scripts/final/binary2visable.py --path $parent_dir/data/ISIC2017/output/test/nnUNet

        # echo "Evaluating on PH2 test..."
        # rm -rf $parent_dir/data/PH2/output/test/nnUNet
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id nnUNetv2_predict -i $parent_dir/data/nnUNetFrame/PH2_test/PredictDataset/imagesTr -o $parent_dir/data/PH2/output/test/nnUNet -d 666 -c 2d -f 0
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/PH2/output/test/nnUNet_timecost.txt
        # python $parent_dir/code/py_scripts/final/binary2visable.py --path $parent_dir/data/PH2/output/test/nnUNet

        # echo "Evaluating on Dermofit test..."
        # rm -rf $parent_dir/data/Dermofit/output/test/nnUNet
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id nnUNetv2_predict -i $parent_dir/data/nnUNetFrame/Dermofit_test/PredictDataset/imagesTr -o $parent_dir/data/Dermofit/output/test/nnUNet -d 666 -c 2d -f 0
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/Dermofit/output/test/nnUNet_timecost.txt
        # python $parent_dir/code/py_scripts/final/binary2visable.py --path $parent_dir/data/Dermofit/output/test/nnUNet

        # echo "Evaluating on Atlas test..."
        # rm -rf $parent_dir/data/Atlas/output/test/nnUNet
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id nnUNetv2_predict -i $parent_dir/data/nnUNetFrame/Atlas_test/PredictDataset/imagesTr -o $parent_dir/data/Atlas/output/test/nnUNet -d 666 -c 2d -f 0
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/Atlas/output/test/nnUNet_timecost.txt
        # python $parent_dir/code/py_scripts/final/binary2visable.py --path $parent_dir/data/Atlas/output/test/nnUNet


        echo "Evaluating on AtlasZoomIn10 test..."
        rm -rf $parent_dir/data/AtlasZoomIn10/output/test/nnUNet
        start_time=$(date +%s)
        CUDA_VISIBLE_DEVICES=$gpu_id nnUNetv2_predict -i $parent_dir/data/nnUNetFrame/AtlasZoomIn10_test/PredictDataset/imagesTr -o $parent_dir/data/AtlasZoomIn10/output/test/nnUNet -d 666 -c 2d -f 0
        end_time=$(date +%s)
        cost_time=$((end_time - start_time))
        echo "Total cost [$cost_time] seconds" > $parent_dir/data/AtlasZoomIn10/output/test/nnUNet_timecost.txt
        python $parent_dir/code/py_scripts/final/binary2visable.py --path $parent_dir/data/AtlasZoomIn10/output/test/nnUNet

        echo "Evaluating on AtlasZoomIn35 test..."
        rm -rf $parent_dir/data/AtlasZoomIn35/output/test/nnUNet
        start_time=$(date +%s)
        CUDA_VISIBLE_DEVICES=$gpu_id nnUNetv2_predict -i $parent_dir/data/nnUNetFrame/AtlasZoomIn35_test/PredictDataset/imagesTr -o $parent_dir/data/AtlasZoomIn35/output/test/nnUNet -d 666 -c 2d -f 0
        end_time=$(date +%s)
        cost_time=$((end_time - start_time))
        echo "Total cost [$cost_time] seconds" > $parent_dir/data/AtlasZoomIn35/output/test/nnUNet_timecost.txt
        python $parent_dir/code/py_scripts/final/binary2visable.py --path $parent_dir/data/AtlasZoomIn35/output/test/nnUNet



        # Return name back.
        echo "Renaming the nnUNet predict files..."
        cd $parent_dir/code/compare_models/
        # python nnUNet-predict_rename_back.py --processing-path $parent_dir/data/HAM10000/output/val/nnUNet --csv-path $parent_dir/data/nnUNetFrame/HAM10000_val/image_file_mapping.csv
        # python nnUNet-predict_rename_back.py --processing-path $parent_dir/data/HAM10000/output/test/nnUNet --csv-path $parent_dir/data/nnUNetFrame/HAM10000_test/image_file_mapping.csv
        # python nnUNet-predict_rename_back.py --processing-path $parent_dir/data/ISIC2016/output/test/nnUNet --csv-path $parent_dir/data/nnUNetFrame/ISIC2016_test/image_file_mapping.csv
        # python nnUNet-predict_rename_back.py --processing-path $parent_dir/data/ISIC2017/output/test/nnUNet --csv-path $parent_dir/data/nnUNetFrame/ISIC2017_test/image_file_mapping.csv
        # python nnUNet-predict_rename_back.py --processing-path $parent_dir/data/PH2/output/test/nnUNet --csv-path $parent_dir/data/nnUNetFrame/PH2_test/image_file_mapping.csv
        # python nnUNet-predict_rename_back.py --processing-path $parent_dir/data/Dermofit/output/test/nnUNet --csv-path $parent_dir/data/nnUNetFrame/Dermofit_test/image_file_mapping.csv
        # python nnUNet-predict_rename_back.py --processing-path $parent_dir/data/Atlas/output/test/nnUNet --csv-path $parent_dir/data/nnUNetFrame/Atlas_test/image_file_mapping.csv
        python nnUNet-predict_rename_back.py --processing-path $parent_dir/data/AtlasZoomIn10/output/test/nnUNet --csv-path $parent_dir/data/nnUNetFrame/AtlasZoomIn10_test/image_file_mapping.csv
        python nnUNet-predict_rename_back.py --processing-path $parent_dir/data/AtlasZoomIn35/output/test/nnUNet --csv-path $parent_dir/data/nnUNetFrame/AtlasZoomIn35_test/image_file_mapping.csv
        
        echo "Renaming the ground truth files..."
        cd $parent_dir/code/compare_models/
        # python nnUNet-predict_rename_back.py --processing-path $parent_dir/data/nnUNetFrame/HAM10000_val/output --csv-path $parent_dir/data/nnUNetFrame/HAM10000_val/image_file_mapping.csv
        # python nnUNet-predict_rename_back.py --processing-path $parent_dir/data/nnUNetFrame/HAM10000_test/output --csv-path $parent_dir/data/nnUNetFrame/HAM10000_test/image_file_mapping.csv
        # python nnUNet-predict_rename_back.py --processing-path $parent_dir/data/nnUNetFrame/ISIC2016_test/output --csv-path $parent_dir/data/nnUNetFrame/ISIC2016_test/image_file_mapping.csv
        # python nnUNet-predict_rename_back.py --processing-path $parent_dir/data/nnUNetFrame/ISIC2017_test/output --csv-path $parent_dir/data/nnUNetFrame/ISIC2017_test/image_file_mapping.csv
        # python nnUNet-predict_rename_back.py --processing-path $parent_dir/data/nnUNetFrame/PH2_test/output --csv-path $parent_dir/data/nnUNetFrame/PH2_test/image_file_mapping.csv
        # python nnUNet-predict_rename_back.py --processing-path $parent_dir/data/nnUNetFrame/Dermofit_test/output --csv-path $parent_dir/data/nnUNetFrame/Dermofit_test/image_file_mapping.csv
        # python nnUNet-predict_rename_back.py --processing-path $parent_dir/data/nnUNetFrame/Atlas_test/output --csv-path $parent_dir/data/nnUNetFrame/Atlas_test/image_file_mapping.csv
        python nnUNet-predict_rename_back.py --processing-path $parent_dir/data/nnUNetFrame/AtlasZoomIn10_test/output --csv-path $parent_dir/data/nnUNetFrame/AtlasZoomIn10_test/image_file_mapping.csv
        python nnUNet-predict_rename_back.py --processing-path $parent_dir/data/nnUNetFrame/AtlasZoomIn35_test/output --csv-path $parent_dir/data/nnUNetFrame/AtlasZoomIn35_test/image_file_mapping.csv

        echo "[√] nnUNet Evaluate finished."
    fi


    if [ "$calculate_metrics_nnUNet" = true ]; then
        cd $parent_dir/code/compare_models/
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/nnUNetFrame/HAM10000_val/output --predict-output $parent_dir/data/HAM10000/output/val/nnUNet --csv-output-path $parent_dir/data/HAM10000/output/val/nnUNet_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/nnUNetFrame/HAM10000_test/output --predict-output $parent_dir/data/HAM10000/output/test/nnUNet --csv-output-path $parent_dir/data/HAM10000/output/test/nnUNet_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/nnUNetFrame/ISIC2016_test/output --predict-output $parent_dir/data/ISIC2016/output/test/nnUNet --csv-output-path $parent_dir/data/ISIC2016/output/test/nnUNet_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/nnUNetFrame/ISIC2017_test/output --predict-output $parent_dir/data/ISIC2017/output/test/nnUNet --csv-output-path $parent_dir/data/ISIC2017/output/test/nnUNet_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/nnUNetFrame/PH2_test/output --predict-output $parent_dir/data/PH2/output/test/nnUNet --csv-output-path $parent_dir/data/PH2/output/test/nnUNet_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/nnUNetFrame/Dermofit_test/output --predict-output $parent_dir/data/Dermofit/output/test/nnUNet --csv-output-path $parent_dir/data/Dermofit/output/test/nnUNet_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/nnUNetFrame/Atlas_test/output --predict-output $parent_dir/data/Atlas/output/test/nnUNet --csv-output-path $parent_dir/data/Atlas/output/test/nnUNet_metrics.csv --image-size $img_resize $img_resize
        python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/nnUNetFrame/AtlasZoomIn10_test/output --predict-output $parent_dir/data/AtlasZoomIn10/output/test/nnUNet --csv-output-path $parent_dir/data/AtlasZoomIn10/output/test/nnUNet_metrics.csv --image-size $img_resize $img_resize
        python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/nnUNetFrame/AtlasZoomIn35_test/output --predict-output $parent_dir/data/AtlasZoomIn35/output/test/nnUNet --csv-output-path $parent_dir/data/AtlasZoomIn35/output/test/nnUNet_metrics.csv --image-size $img_resize $img_resize
    fi

    if [ "$upload_nnUNet" = true ]; then
        echo "Checking nnUNet output..."
        cd $parent_dir/code
        # python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_val_img --data-path $parent_dir/data/HAM10000/output/val/nnUNet/ --image-size $img_resize $img_resize --image-num 1001
        # python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_test_img --data-path $parent_dir/data/HAM10000/output/test/nnUNet/ --image-size $img_resize $img_resize --image-num 1002
        # python py_scripts/preprocessing/check_image_size.py --data-name ISIC2016_test_img --data-path $parent_dir/data/ISIC2016/output/test/nnUNet/ --image-size $img_resize $img_resize --image-num 379
        # python py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_test_img --data-path $parent_dir/data/ISIC2017/output/test/nnUNet/ --image-size $img_resize $img_resize --image-num 600
        # python py_scripts/preprocessing/check_image_size.py --data-name PH2_test_img --data-path $parent_dir/data/PH2/output/test/nnUNet/ --image-size $img_resize $img_resize --image-num 200
        # python py_scripts/preprocessing/check_image_size.py --data-name Dermofit_test_img --data-path $parent_dir/data/Dermofit/output/test/nnUNet/ --image-size $img_resize $img_resize --image-num 1300
        # python py_scripts/preprocessing/check_image_size.py --data-name Atlas_test_img --data-path $parent_dir/data/Atlas/output/test/nnUNet/ --image-size $img_resize $img_resize --image-num 960
        python py_scripts/preprocessing/check_image_size.py --data-name AtlasZoomIn10_test_img --data-path $parent_dir/data/AtlasZoomIn10/output/test/nnUNet/ --image-size $img_resize $img_resize --image-num 960
        python py_scripts/preprocessing/check_image_size.py --data-name AtlasZoomIn35_test_img --data-path $parent_dir/data/AtlasZoomIn35/output/test/nnUNet/ --image-size $img_resize $img_resize --image-num 960
        echo "[√] nnUNet output check pass."


        cd $parent_dir/code/compare_models/
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name nnUNet-HAM10000_val --metrics-csv $parent_dir/data/HAM10000/output/val/nnUNet_metrics.csv --timecost-txt $parent_dir/data/HAM10000/output/val/nnUNet_timecost.txt --images-num 1001
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name nnUNet-HAM10000_test --metrics-csv $parent_dir/data/HAM10000/output/test/nnUNet_metrics.csv --timecost-txt $parent_dir/data/HAM10000/output/test/nnUNet_timecost.txt --images-num 1002
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name nnUNet-ISIC2016_test --metrics-csv $parent_dir/data/ISIC2016/output/test/nnUNet_metrics.csv --timecost-txt $parent_dir/data/ISIC2016/output/test/nnUNet_timecost.txt --images-num 379
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name nnUNet-ISIC2017_test --metrics-csv $parent_dir/data/ISIC2017/output/test/nnUNet_metrics.csv --timecost-txt $parent_dir/data/ISIC2017/output/test/nnUNet_timecost.txt --images-num 600
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name nnUNet-PH2_test --metrics-csv $parent_dir/data/PH2/output/test/nnUNet_metrics.csv --timecost-txt $parent_dir/data/PH2/output/test/nnUNet_timecost.txt --images-num 200
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name nnUNet-Dermofit_test --metrics-csv $parent_dir/data/Dermofit/output/test/nnUNet_metrics.csv --timecost-txt $parent_dir/data/Dermofit/output/test/nnUNet_timecost.txt --images-num 1300
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name nnUNet-Atlas_test --metrics-csv $parent_dir/data/Atlas/output/test/nnUNet_metrics.csv --timecost-txt $parent_dir/data/Atlas/output/test/nnUNet_timecost.txt --images-num 960
        python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name nnUNet-AtlasZoomIn10_test --metrics-csv $parent_dir/data/AtlasZoomIn10/output/test/nnUNet_metrics.csv --timecost-txt $parent_dir/data/AtlasZoomIn10/output/test/nnUNet_timecost.txt --images-num 960
        python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name nnUNet-AtlasZoomIn35_test --metrics-csv $parent_dir/data/AtlasZoomIn35/output/test/nnUNet_metrics.csv --timecost-txt $parent_dir/data/AtlasZoomIn35/output/test/nnUNet_timecost.txt --images-num 960
        echo "[√] nnUNet Metrics upload finished."
    fi


    cd $parent_dir/code
    echo "✅ nnUNet compare finished."
    echo "#########################################################################"
fi

# ############################
# # SAM Auto                ##
# ############################
if [ "$COMPARE_SAMAUTO" = true ]; then
    echo "Comparing with SAMAuto..."
    if [ "$evaluate_SAMAuto" = true ]; then
        echo "Evaluating on SAMAuto..."
        cd $parent_dir/code/
        # python ./compare_models/SAMAuto_predict.py --input-path $parent_dir/data/HAM10000/input/val/HAM10000_img --gt-path $parent_dir/data/HAM10000/input/val/HAM10000_seg --output-path $parent_dir/data/HAM10000/output/val/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth
        # python ./compare_models/SAMAuto_predict.py --input-path $parent_dir/data/HAM10000/input/test/HAM10000_img --gt-path $parent_dir/data/HAM10000/input/test/HAM10000_seg --output-path $parent_dir/data/HAM10000/output/test/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth
        # python ./compare_models/SAMAuto_predict.py --input-path $parent_dir/data/ISIC2016/input/test/ISIC2016_img --gt-path $parent_dir/data/ISIC2016/input/test/ISIC2016_seg --output-path $parent_dir/data/ISIC2016/output/test/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth
        # python ./compare_models/SAMAuto_predict.py --input-path $parent_dir/data/ISIC2017/input/test/ISIC2017_img --gt-path $parent_dir/data/ISIC2017/input/test/ISIC2017_seg --output-path $parent_dir/data/ISIC2017/output/test/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth
        # python ./compare_models/SAMAuto_predict.py --input-path $parent_dir/data/PH2/input/test/PH2_img --gt-path $parent_dir/data/PH2/input/test/PH2_seg --output-path $parent_dir/data/PH2/output/test/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth
        # python ./compare_models/SAMAuto_predict.py --input-path $parent_dir/data/Dermofit/input/test/Dermofit_img --gt-path $parent_dir/data/Dermofit/input/test/Dermofit_seg --output-path $parent_dir/data/Dermofit/output/test/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth
        # python ./compare_models/SAMAuto_predict.py --input-path $parent_dir/data/Atlas/input/test/Atlas_img --gt-path $parent_dir/data/Atlas/input/test/Atlas_seg --output-path $parent_dir/data/Atlas/output/test/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth
        python ./compare_models/SAMAuto_predict.py --input-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_img --gt-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_seg --output-path $parent_dir/data/AtlasZoomIn10/output/test/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth
    fi

    if [ "$calculate_metrics_SAMAuto" = true ]; then

        echo "Calculate SAMAuto_timecost from csv..."
        cd $parent_dir/code/py_scripts/final
        # python calculate_timecost.py --input-csv-path $parent_dir/data/HAM10000/output/val/SAMAuto_timecost.csv --output-txt-path $parent_dir/data/HAM10000/output/val/SAMAuto_timecost.txt --images-num 1001
        # python calculate_timecost.py --input-csv-path $parent_dir/data/HAM10000/output/test/SAMAuto_timecost.csv --output-txt-path $parent_dir/data/HAM10000/output/test/SAMAuto_timecost.txt --images-num 1002
        # python calculate_timecost.py --input-csv-path $parent_dir/data/ISIC2016/output/test/SAMAuto_timecost.csv --output-txt-path $parent_dir/data/ISIC2016/output/test/SAMAuto_timecost.txt --images-num 379
        # python calculate_timecost.py --input-csv-path $parent_dir/data/ISIC2017/output/test/SAMAuto_timecost.csv --output-txt-path $parent_dir/data/ISIC2017/output/test/SAMAuto_timecost.txt --images-num 600
        # python calculate_timecost.py --input-csv-path $parent_dir/data/PH2/output/test/SAMAuto_timecost.csv --output-txt-path $parent_dir/data/PH2/output/test/SAMAuto_timecost.txt --images-num 200
        # python calculate_timecost.py --input-csv-path $parent_dir/data/Dermofit/output/test/SAMAuto_timecost.csv --output-txt-path $parent_dir/data/Dermofit/output/test/SAMAuto_timecost.txt --images-num 1300
        # python calculate_timecost.py --input-csv-path $parent_dir/data/Atlas/output/test/SAMAuto_timecost.csv --output-txt-path $parent_dir/data/Atlas/output/test/SAMAuto_timecost.txt --images-num 960
        python calculate_timecost.py --input-csv-path $parent_dir/data/AtlasZoomIn10/output/test/SAMAuto_timecost.csv --output-txt-path $parent_dir/data/AtlasZoomIn10/output/test/SAMAuto_timecost.txt --images-num 960
        echo "[√] SAMAuto_timecost calculation finish."

        cd $parent_dir/code/compare_models/
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/HAM10000/input/val/HAM10000_seg --predict-output $parent_dir/data/HAM10000/output/val/SAMAuto --csv-output-path $parent_dir/data/HAM10000/output/val/SAMAuto_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/HAM10000/input/test/HAM10000_seg --predict-output $parent_dir/data/HAM10000/output/test/SAMAuto --csv-output-path $parent_dir/data/HAM10000/output/test/SAMAuto_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/ISIC2016/input/test/ISIC2016_seg --predict-output $parent_dir/data/ISIC2016/output/test/SAMAuto --csv-output-path $parent_dir/data/ISIC2016/output/test/SAMAuto_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/ISIC2017/input/test/ISIC2017_seg --predict-output $parent_dir/data/ISIC2017/output/test/SAMAuto --csv-output-path $parent_dir/data/ISIC2017/output/test/SAMAuto_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/PH2/input/test/PH2_seg --predict-output $parent_dir/data/PH2/output/test/SAMAuto --csv-output-path $parent_dir/data/PH2/output/test/SAMAuto_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/Dermofit/input/test/Dermofit_seg --predict-output $parent_dir/data/Dermofit/output/test/SAMAuto --csv-output-path $parent_dir/data/Dermofit/output/test/SAMAuto_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/Atlas/input/test/Atlas_seg --predict-output $parent_dir/data/Atlas/output/test/SAMAuto --csv-output-path $parent_dir/data/Atlas/output/test/SAMAuto_metrics.csv --image-size $img_resize $img_resize
        python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_seg --predict-output $parent_dir/data/AtlasZoomIn10/output/test/SAMAuto --csv-output-path $parent_dir/data/AtlasZoomIn10/output/test/SAMAuto_metrics.csv --image-size $img_resize $img_resize
        echo "[√] SAMAuto Metrics calculation finish."
    fi

    if [ "$upload_SAMAuto" = true ]; then

        echo "Checking SAMAuto output..."
        cd $parent_dir/code
        # python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_val_img --data-path $parent_dir/data/HAM10000/output/val/SAMAuto/ --image-size $img_resize $img_resize --image-num 1001
        # python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_test_img --data-path $parent_dir/data/HAM10000/output/test/SAMAuto/ --image-size $img_resize $img_resize --image-num 1002
        # python py_scripts/preprocessing/check_image_size.py --data-name ISIC2016_test_img --data-path $parent_dir/data/ISIC2016/output/test/SAMAuto/ --image-size $img_resize $img_resize --image-num 379
        # python py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_test_img --data-path $parent_dir/data/ISIC2017/output/test/SAMAuto/ --image-size $img_resize $img_resize --image-num 600
        # python py_scripts/preprocessing/check_image_size.py --data-name PH2_test_img --data-path $parent_dir/data/PH2/output/test/SAMAuto/ --image-size $img_resize $img_resize --image-num 200
        # python py_scripts/preprocessing/check_image_size.py --data-name Dermofit_test_img --data-path $parent_dir/data/Dermofit/output/test/SAMAuto/ --image-size $img_resize $img_resize --image-num 1300
        # python py_scripts/preprocessing/check_image_size.py --data-name Atlas_test_img --data-path $parent_dir/data/Atlas/output/test/SAMAuto/ --image-size $img_resize $img_resize --image-num 960
        python py_scripts/preprocessing/check_image_size.py --data-name AtlasZoomIn10_test_img --data-path $parent_dir/data/AtlasZoomIn10/output/test/SAMAuto/ --image-size $img_resize $img_resize --image-num 960
        echo "[√] SAMAuto output check pass."

        cd $parent_dir/code/compare_models/
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMAuto-HAM10000_val --metrics-csv $parent_dir/data/HAM10000/output/val/SAMAuto_metrics.csv --timecost-txt $parent_dir/data/HAM10000/output/val/SAMAuto_timecost.txt --images-num 1001
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMAuto-HAM10000_test --metrics-csv $parent_dir/data/HAM10000/output/test/SAMAuto_metrics.csv --timecost-txt $parent_dir/data/HAM10000/output/test/SAMAuto_timecost.txt --images-num 1002
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMAuto-ISIC2016_test --metrics-csv $parent_dir/data/ISIC2016/output/test/SAMAuto_metrics.csv --timecost-txt $parent_dir/data/ISIC2016/output/test/SAMAuto_timecost.txt --images-num 379
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMAuto-ISIC2017_test --metrics-csv $parent_dir/data/ISIC2017/output/test/SAMAuto_metrics.csv --timecost-txt $parent_dir/data/ISIC2017/output/test/SAMAuto_timecost.txt --images-num 600
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMAuto-PH2_test --metrics-csv $parent_dir/data/PH2/output/test/SAMAuto_metrics.csv --timecost-txt $parent_dir/data/PH2/output/test/SAMAuto_timecost.txt --images-num 200
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMAuto-Dermofit_test --metrics-csv $parent_dir/data/Dermofit/output/test/SAMAuto_metrics.csv --timecost-txt $parent_dir/data/Dermofit/output/test/SAMAuto_timecost.txt --images-num 1300
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMAuto-Atlas_test --metrics-csv $parent_dir/data/Atlas/output/test/SAMAuto_metrics.csv --timecost-txt $parent_dir/data/Atlas/output/test/SAMAuto_timecost.txt --images-num 960
        python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMAuto-AtlasZoomIn10_test --metrics-csv $parent_dir/data/AtlasZoomIn10/output/test/SAMAuto_metrics.csv --timecost-txt $parent_dir/data/AtlasZoomIn10/output/test/SAMAuto_timecost.txt --images-num 960
        echo "[√] SAMAuto Metrics upload finished."
    fi

fi


# ############################
# #  SAM Specific Points    ##
# ############################
if [ "$COMPARE_SAMSPECIFIC" = true ]; then
    echo "Comparing with SAM specific points..."
    if [ "$evaluate_SAMSpecific" = true ]; then
        echo "Evaluating on SAMSpecific..."
        cd $parent_dir/code/
        # python ./compare_models/SAMSpecific_predict.py --input-path $parent_dir/data/HAM10000/input/val/HAM10000_img --gt-path $parent_dir/data/HAM10000/input/val/HAM10000_seg --output-path $parent_dir/data/HAM10000/output/val/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth --point-csv-path $parent_dir/data/HAM10000/input/HAM10000_val_prompts_$labeller.csv
        # python ./compare_models/SAMSpecific_predict.py --input-path $parent_dir/data/HAM10000/input/test/HAM10000_img --gt-path $parent_dir/data/HAM10000/input/test/HAM10000_seg --output-path $parent_dir/data/HAM10000/output/test/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth --point-csv-path $parent_dir/data/HAM10000/input/HAM10000_test_prompts_$labeller.csv
        # python ./compare_models/SAMSpecific_predict.py --input-path $parent_dir/data/ISIC2016/input/test/ISIC2016_img --gt-path $parent_dir/data/ISIC2016/input/test/ISIC2016_seg --output-path $parent_dir/data/ISIC2016/output/test/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth --point-csv-path $parent_dir/data/ISIC2016/input/ISIC2016_test_prompts_$labeller.csv
        # python ./compare_models/SAMSpecific_predict.py --input-path $parent_dir/data/ISIC2017/input/test/ISIC2017_img --gt-path $parent_dir/data/ISIC2017/input/test/ISIC2017_seg --output-path $parent_dir/data/ISIC2017/output/test/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth --point-csv-path $parent_dir/data/ISIC2017/input/ISIC2017_test_prompts_$labeller.csv
        # python ./compare_models/SAMSpecific_predict.py --input-path $parent_dir/data/PH2/input/test/PH2_img --gt-path $parent_dir/data/PH2/input/test/PH2_seg --output-path $parent_dir/data/PH2/output/test/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth --point-csv-path $parent_dir/data/PH2/input/PH2_test_prompts_$labeller.csv
        # python ./compare_models/SAMSpecific_predict.py --input-path $parent_dir/data/Dermofit/input/test/Dermofit_img --gt-path $parent_dir/data/Dermofit/input/test/Dermofit_seg --output-path $parent_dir/data/Dermofit/output/test/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth --point-csv-path $parent_dir/data/Dermofit/input/Dermofit_test_prompts_$labeller.csv
        # python ./compare_models/SAMSpecific_predict.py --input-path $parent_dir/data/Atlas/input/test/Atlas_img --gt-path $parent_dir/data/Atlas/input/test/Atlas_seg --output-path $parent_dir/data/Atlas/output/test/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth --point-csv-path $parent_dir/data/Atlas/input/Atlas_test_prompts_$labeller.csv
        python ./compare_models/SAMSpecific_predict.py --input-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_img --gt-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_seg --output-path $parent_dir/data/AtlasZoomIn10/output/test/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth --point-csv-path $parent_dir/data/AtlasZoomIn10/input/AtlasZoomIn10_test_prompts_$labeller.csv
        python ./compare_models/SAMSpecific_predict.py --input-path $parent_dir/data/AtlasZoomIn35/input/test/AtlasZoomIn35_img --gt-path $parent_dir/data/AtlasZoomIn35/input/test/AtlasZoomIn35_seg --output-path $parent_dir/data/AtlasZoomIn35/output/test/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth --point-csv-path $parent_dir/data/AtlasZoomIn35/input/AtlasZoomIn35_test_prompts_$labeller.csv
    fi

    if [ "$calculate_metrics_SAMSpecific" = true ]; then
        echo "Calculate SAMAuto_timecost from csv..."
        cd $parent_dir/code/py_scripts/final
        # python calculate_timecost.py --input-csv-path $parent_dir/data/HAM10000/output/val/SAMSpecific_timecost.csv --output-txt-path $parent_dir/data/HAM10000/output/val/SAMSpecific_timecost.txt --images-num 1001
        # python calculate_timecost.py --input-csv-path $parent_dir/data/HAM10000/output/test/SAMSpecific_timecost.csv --output-txt-path $parent_dir/data/HAM10000/output/test/SAMSpecific_timecost.txt --images-num 1002
        # python calculate_timecost.py --input-csv-path $parent_dir/data/ISIC2016/output/test/SAMSpecific_timecost.csv --output-txt-path $parent_dir/data/ISIC2016/output/test/SAMSpecific_timecost.txt --images-num 379
        # python calculate_timecost.py --input-csv-path $parent_dir/data/ISIC2017/output/test/SAMSpecific_timecost.csv --output-txt-path $parent_dir/data/ISIC2017/output/test/SAMSpecific_timecost.txt --images-num 600
        # python calculate_timecost.py --input-csv-path $parent_dir/data/PH2/output/test/SAMSpecific_timecost.csv --output-txt-path $parent_dir/data/PH2/output/test/SAMSpecific_timecost.txt --images-num 200
        # python calculate_timecost.py --input-csv-path $parent_dir/data/Dermofit/output/test/SAMSpecific_timecost.csv --output-txt-path $parent_dir/data/Dermofit/output/test/SAMSpecific_timecost.txt --images-num 1300
        # python calculate_timecost.py --input-csv-path $parent_dir/data/Atlas/output/test/SAMSpecific_timecost.csv --output-txt-path $parent_dir/data/Atlas/output/test/SAMSpecific_timecost.txt --images-num 960
        python calculate_timecost.py --input-csv-path $parent_dir/data/AtlasZoomIn10/output/test/SAMSpecific_timecost.csv --output-txt-path $parent_dir/data/AtlasZoomIn10/output/test/SAMSpecific_timecost.txt --images-num 960
        python calculate_timecost.py --input-csv-path $parent_dir/data/AtlasZoomIn35/output/test/SAMSpecific_timecost.csv --output-txt-path $parent_dir/data/AtlasZoomIn35/output/test/SAMSpecific_timecost.txt --images-num 960
        echo "[√] SAMSpecific_timecost calculation finish."

        cd $parent_dir/code/compare_models/
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/HAM10000/input/val/HAM10000_seg --predict-output $parent_dir/data/HAM10000/output/val/SAMSpecific --csv-output-path $parent_dir/data/HAM10000/output/val/SAMSpecific_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/HAM10000/input/test/HAM10000_seg --predict-output $parent_dir/data/HAM10000/output/test/SAMSpecific --csv-output-path $parent_dir/data/HAM10000/output/test/SAMSpecific_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/ISIC2016/input/test/ISIC2016_seg --predict-output $parent_dir/data/ISIC2016/output/test/SAMSpecific --csv-output-path $parent_dir/data/ISIC2016/output/test/SAMSpecific_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/ISIC2017/input/test/ISIC2017_seg --predict-output $parent_dir/data/ISIC2017/output/test/SAMSpecific --csv-output-path $parent_dir/data/ISIC2017/output/test/SAMSpecific_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/PH2/input/test/PH2_seg --predict-output $parent_dir/data/PH2/output/test/SAMSpecific --csv-output-path $parent_dir/data/PH2/output/test/SAMSpecific_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/Dermofit/input/test/Dermofit_seg --predict-output $parent_dir/data/Dermofit/output/test/SAMSpecific --csv-output-path $parent_dir/data/Dermofit/output/test/SAMSpecific_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/Atlas/input/test/Atlas_seg --predict-output $parent_dir/data/Atlas/output/test/SAMSpecific --csv-output-path $parent_dir/data/Atlas/output/test/SAMSpecific_metrics.csv --image-size $img_resize $img_resize
        python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_seg --predict-output $parent_dir/data/AtlasZoomIn10/output/test/SAMSpecific --csv-output-path $parent_dir/data/AtlasZoomIn10/output/test/SAMSpecific_metrics.csv --image-size $img_resize $img_resize
        python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/AtlasZoomIn35/input/test/AtlasZoomIn35_seg --predict-output $parent_dir/data/AtlasZoomIn35/output/test/SAMSpecific --csv-output-path $parent_dir/data/AtlasZoomIn35/output/test/SAMSpecific_metrics.csv --image-size $img_resize $img_resize
        echo "[√] SAMSpecific Metrics calculation finish."
    fi

    if [ "$upload_SAMSpecific" = true ]; then
        echo "Checking SAMSpecific output..."
        cd $parent_dir/code
        # python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_val_img --data-path $parent_dir/data/HAM10000/output/val/SAMSpecific/ --image-size $img_resize $img_resize --image-num 1001
        # python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_test_img --data-path $parent_dir/data/HAM10000/output/test/SAMSpecific/ --image-size $img_resize $img_resize --image-num 1002
        # python py_scripts/preprocessing/check_image_size.py --data-name ISIC2016_test_img --data-path $parent_dir/data/ISIC2016/output/test/SAMSpecific/ --image-size $img_resize $img_resize --image-num 379
        # python py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_test_img --data-path $parent_dir/data/ISIC2017/output/test/SAMSpecific/ --image-size $img_resize $img_resize --image-num 600
        # python py_scripts/preprocessing/check_image_size.py --data-name PH2_test_img --data-path $parent_dir/data/PH2/output/test/SAMSpecific/ --image-size $img_resize $img_resize --image-num 200
        # python py_scripts/preprocessing/check_image_size.py --data-name Dermofit_test_img --data-path $parent_dir/data/Dermofit/output/test/SAMSpecific/ --image-size $img_resize $img_resize --image-num 1300
        # python py_scripts/preprocessing/check_image_size.py --data-name Atlas_test_img --data-path $parent_dir/data/Atlas/output/test/SAMSpecific/ --image-size $img_resize $img_resize --image-num 960
        python py_scripts/preprocessing/check_image_size.py --data-name AtlasZoomIn10_test_img --data-path $parent_dir/data/AtlasZoomIn10/output/test/SAMSpecific/ --image-size $img_resize $img_resize --image-num 960
        python py_scripts/preprocessing/check_image_size.py --data-name AtlasZoomIn35_test_img --data-path $parent_dir/data/AtlasZoomIn35/output/test/SAMSpecific/ --image-size $img_resize $img_resize --image-num 960
        echo "[√] SAMSpecific output check pass."

        cd $parent_dir/code/compare_models/
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMSpecific-HAM10000_val --metrics-csv $parent_dir/data/HAM10000/output/val/SAMSpecific_metrics.csv --timecost-txt $parent_dir/data/HAM10000/output/val/SAMSpecific_timecost.txt --images-num 1001
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMSpecific-HAM10000_test --metrics-csv $parent_dir/data/HAM10000/output/test/SAMSpecific_metrics.csv --timecost-txt $parent_dir/data/HAM10000/output/test/SAMSpecific_timecost.txt --images-num 1002
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMSpecific-ISIC2016_test --metrics-csv $parent_dir/data/ISIC2016/output/test/SAMSpecific_metrics.csv --timecost-txt $parent_dir/data/ISIC2016/output/test/SAMSpecific_timecost.txt --images-num 379
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMSpecific-ISIC2017_test --metrics-csv $parent_dir/data/ISIC2017/output/test/SAMSpecific_metrics.csv --timecost-txt $parent_dir/data/ISIC2017/output/test/SAMSpecific_timecost.txt --images-num 600
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMSpecific-PH2_test --metrics-csv $parent_dir/data/PH2/output/test/SAMSpecific_metrics.csv --timecost-txt $parent_dir/data/PH2/output/test/SAMSpecific_timecost.txt --images-num 200
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMSpecific-Dermofit_test --metrics-csv $parent_dir/data/Dermofit/output/test/SAMSpecific_metrics.csv --timecost-txt $parent_dir/data/Dermofit/output/test/SAMSpecific_timecost.txt --images-num 1300
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMSpecific-Atlas_test --metrics-csv $parent_dir/data/Atlas/output/test/SAMSpecific_metrics.csv --timecost-txt $parent_dir/data/Atlas/output/test/SAMSpecific_timecost.txt --images-num 960
        python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMSpecific-AtlasZoomIn10_test --metrics-csv $parent_dir/data/AtlasZoomIn10/output/test/SAMSpecific_metrics.csv --timecost-txt $parent_dir/data/AtlasZoomIn10/output/test/SAMSpecific_timecost.txt --images-num 960
        python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMSpecific-AtlasZoomIn35_test --metrics-csv $parent_dir/data/AtlasZoomIn35/output/test/SAMSpecific_metrics.csv --timecost-txt $parent_dir/data/AtlasZoomIn35/output/test/SAMSpecific_timecost.txt --images-num 960
        echo "[√] SAMSpecific Metrics upload finished."
    fi
fi


############################
#  OnePrompt              ##
############################
if [ "$COMPARE_ONEPROMPT" = true ]; then
    echo "Comparing with OnePrompt..."
    mkdir -p $parent_dir/code/compare_models/reps/
    cd $parent_dir/code/compare_models/reps/
    if [ -d "OnePrompt" ];
    then
        echo "[√] OnePrompt already exists."
    else
        git clone https://github.com/KidsWithTokens/one-prompt.git OnePrompt
        cd $parent_dir/code/compare_models/reps/OnePrompt
        # pip install -e .
    fi

    if [ "$retrain_OnePrompt" = true ]; then
        cd $parent_dir/code/py_scripts/preprocessing/
        # HAM10000
        python generate_monai_json.py --dataset-name HAM10000 --path $parent_dir/data/HAM10000/input
        python generate_dataset_csv.py --img-path $parent_dir/data/HAM10000/input/train/HAM10000_img --seg-path $parent_dir/data/HAM10000/input/train/HAM10000_seg --csv-path $parent_dir/data/HAM10000/input/HAM10000_train.csv
        python generate_dataset_csv.py --img-path $parent_dir/data/HAM10000/input/val/HAM10000_img --seg-path $parent_dir/data/HAM10000/input/val/HAM10000_seg --csv-path $parent_dir/data/HAM10000/input/HAM10000_val.csv
        python generate_dataset_csv.py --img-path $parent_dir/data/HAM10000/input/test/HAM10000_img --seg-path $parent_dir/data/HAM10000/input/test/HAM10000_seg --csv-path $parent_dir/data/HAM10000/input/HAM10000_test.csv
        
        # Other test dataset
        python generate_dataset_csv.py --img-path $parent_dir/data/ISIC2016/input/test/ISIC2016_img --seg-path $parent_dir/data/ISIC2016/input/test/ISIC2016_seg --csv-path $parent_dir/data/ISIC2016/input/ISIC2016_test.csv
        python generate_dataset_csv.py --img-path $parent_dir/data/ISIC2017/input/test/ISIC2017_img --seg-path $parent_dir/data/ISIC2017/input/test/ISIC2017_seg --csv-path $parent_dir/data/ISIC2017/input/ISIC2017_test.csv
        python generate_dataset_csv.py --img-path $parent_dir/data/PH2/input/test/PH2_img --seg-path $parent_dir/data/PH2/input/test/PH2_seg --csv-path $parent_dir/data/PH2/input/PH2_test.csv
        python generate_dataset_csv.py --img-path $parent_dir/data/Atlas/input/test/Atlas_img --seg-path $parent_dir/data/Atlas/input/test/Atlas_seg --csv-path $parent_dir/data/Atlas/input/Atlas_test.csv
        python generate_dataset_csv.py --img-path $parent_dir/data/Dermofit/input/test/Dermofit_img --seg-path $parent_dir/data/Dermofit/input/test/Dermofit_seg --csv-path $parent_dir/data/Dermofit/input/Dermofit_test.csv


        cd $parent_dir/code/compare_models/reps/OnePrompt
        CUDA_VISIBLE_DEVICES=$gpu_id python train.py -net oneprompt -mod one_adpt -exp_name HAM10000_point -b 1 -dataset HAM10000_point -data_path $parent_dir/data/HAM10000/input -baseline 'unet' -image_size 256 -val_freq 1 -labeller 42
        mv $parent_dir/code/compare_models/reps/OnePrompt/logs/* $parent_dir/code/compare_models/reps/OnePrompt/checkpoint
    fi

    if [ "$evaluate_OnePrompt" = true ]; then
        cd $parent_dir/code/compare_models/reps/OnePrompt

        # 使用 find 命令找到第一个符合条件的目录
        weight_path=$(find $parent_dir/code/compare_models/reps/OnePrompt/checkpoint/ -type d -name 'HAM10000_point_*' | head -n 1)
        weight_path="${weight_path}/Model/checkpoint_best.pth"
        if [ ! -f "$weight_path" ]; then
            echo "❌ Error: No directory found: $weight_path"
            exit 1
        else
            echo "Set OnePrompt weight path: $weight_path"
        fi
        rm -rf $parent_dir/code/compare_models/reps/OnePrompt/logs/*

        # echo "Evaluating on HAM10000 test..."
        # rm -rf $parent_dir/data/HAM10000/output/test/OnePrompt_points
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id python val.py -net oneprompt -mod one_adpt -exp_name HAM10000_point_test -weights $weight_path -b 1 -dataset HAM10000_test -data_path $parent_dir/data/HAM10000/input -output_path $parent_dir/data/HAM10000/output/test/OnePrompt_points -vis 10 -baseline 'unet' -image_size 256 -save_resize $img_resize
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/HAM10000/output/test/OnePrompt_points_timecost.txt

        # echo "Evaluating on HAM10000 val..."
        # rm -rf $parent_dir/data/HAM10000/output/val/OnePrompt_points
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id python val.py -net oneprompt -mod one_adpt -exp_name HAM10000_point_val -weights $weight_path -b 1 -dataset HAM10000_val -data_path $parent_dir/data/HAM10000/input -output_path $parent_dir/data/HAM10000/output/val/OnePrompt_points -vis 10 -baseline 'unet' -image_size 256 -save_resize $img_resize
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/HAM10000/output/val/OnePrompt_points_timecost.txt

        # echo "Evaluating on ISIC2016 test..."
        # rm -rf $parent_dir/data/ISIC2016/output/test/OnePrompt_points
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id python val.py -net oneprompt -mod one_adpt -exp_name ISIC2016_point_test -weights $weight_path -b 1 -dataset OtherTest -csv_path $parent_dir/data/ISIC2016/input/ISIC2016_test.csv -data_path $parent_dir/data/ISIC2016/input -output_path $parent_dir/data/ISIC2016/output/test/OnePrompt_points -vis 10 -baseline 'unet' -image_size 256 -save_resize $img_resize
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/ISIC2016/output/test/OnePrompt_points_timecost.txt

        # echo "Evaluating on ISIC2017 test..."
        # rm -rf $parent_dir/data/ISIC2017/output/test/OnePrompt_points
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id python val.py -net oneprompt -mod one_adpt -exp_name ISIC2017_point_test -weights $weight_path -b 1 -dataset OtherTest -csv_path $parent_dir/data/ISIC2017/input/ISIC2017_test.csv -data_path $parent_dir/data/ISIC2017/input -output_path $parent_dir/data/ISIC2017/output/test/OnePrompt_points -vis 10 -baseline 'unet' -image_size 256 -save_resize $img_resize
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/ISIC2017/output/test/OnePrompt_points_timecost.txt

        # echo "Evaluating on PH2 test..."
        # rm -rf $parent_dir/data/PH2/output/test/OnePrompt_points
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id python val.py -net oneprompt -mod one_adpt -exp_name PH2_point_test -weights $weight_path -b 1 -dataset OtherTest -csv_path $parent_dir/data/PH2/input/PH2_test.csv -data_path $parent_dir/data/PH2/input -output_path $parent_dir/data/PH2/output/test/OnePrompt_points -vis 10 -baseline 'unet' -image_size 256 -save_resize $img_resize
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/PH2/output/test/OnePrompt_points_timecost.txt

        # echo "Evaluating on Atlas test..."
        # rm -rf $parent_dir/data/Atlas/output/test/OnePrompt_points
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id python val.py -net oneprompt -mod one_adpt -exp_name Atlas_point_test -weights $weight_path -b 1 -dataset OtherTest -csv_path $parent_dir/data/Atlas/input/Atlas_test.csv -data_path $parent_dir/data/Atlas/input -output_path $parent_dir/data/Atlas/output/test/OnePrompt_points -vis 10 -baseline 'unet' -image_size 256 -save_resize $img_resize
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/Atlas/output/test/OnePrompt_points_timecost.txt

        # echo "Evaluating on Dermofit test..."
        # rm -rf $parent_dir/data/Dermofit/output/test/OnePrompt_points
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id python val.py -net oneprompt -mod one_adpt -exp_name Dermofit_point_test -weights $weight_path -b 1 -dataset OtherTest -csv_path $parent_dir/data/Dermofit/input/Dermofit_test.csv -data_path $parent_dir/data/Dermofit/input -output_path $parent_dir/data/Dermofit/output/test/OnePrompt_points -vis 10 -baseline 'unet' -image_size 256 -save_resize $img_resize
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/Dermofit/output/test/OnePrompt_points_timecost.txt
    
        cd $parent_dir/code/py_scripts/preprocessing/
        python generate_dataset_csv.py --img-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_img --seg-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_seg --csv-path $parent_dir/data/AtlasZoomIn10/input/AtlasZoomIn10_test.csv

        cd $parent_dir/code/compare_models/reps/OnePrompt
        echo "Evaluating on AtlasZoomIn10 test..."
        rm -rf $parent_dir/data/AtlasZoomIn10/output/test/OnePrompt_points
        start_time=$(date +%s)
        CUDA_VISIBLE_DEVICES=$gpu_id python val.py -net oneprompt -mod one_adpt -exp_name AtlasZoomIn10_point_test -weights $weight_path -b 1 -dataset OtherTest -csv_path $parent_dir/data/AtlasZoomIn10/input/AtlasZoomIn10_test.csv -data_path $parent_dir/data/AtlasZoomIn10/input -output_path $parent_dir/data/AtlasZoomIn10/output/test/OnePrompt_points -vis 10 -baseline 'unet' -image_size 256 -save_resize $img_resize
        end_time=$(date +%s)
        cost_time=$((end_time - start_time))
        echo "Total cost [$cost_time] seconds" > $parent_dir/data/AtlasZoomIn10/output/test/OnePrompt_points_timecost.txt
    
    fi

    if [ "$calculate_metrics_OnePrompt" = true ]; then
        cd $parent_dir/code/compare_models/
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/HAM10000/input/val/HAM10000_seg --predict-output $parent_dir/data/HAM10000/output/val/OnePrompt_points --csv-output-path $parent_dir/data/HAM10000/output/val/OnePrompt_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/HAM10000/input/test/HAM10000_seg --predict-output $parent_dir/data/HAM10000/output/test/OnePrompt_points --csv-output-path $parent_dir/data/HAM10000/output/test/OnePrompt_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/ISIC2016/input/test/ISIC2016_seg --predict-output $parent_dir/data/ISIC2016/output/test/OnePrompt_points --csv-output-path $parent_dir/data/ISIC2016/output/test/OnePrompt_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/ISIC2017/input/test/ISIC2017_seg --predict-output $parent_dir/data/ISIC2017/output/test/OnePrompt_points --csv-output-path $parent_dir/data/ISIC2017/output/test/OnePrompt_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/PH2/input/test/PH2_seg --predict-output $parent_dir/data/PH2/output/test/OnePrompt_points --csv-output-path $parent_dir/data/PH2/output/test/OnePrompt_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/Dermofit/input/test/Dermofit_seg --predict-output $parent_dir/data/Dermofit/output/test/OnePrompt_points --csv-output-path $parent_dir/data/Dermofit/output/test/OnePrompt_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/Atlas/input/test/Atlas_seg --predict-output $parent_dir/data/Atlas/output/test/OnePrompt_points --csv-output-path $parent_dir/data/Atlas/output/test/OnePrompt_metrics.csv --image-size $img_resize $img_resize
        python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_seg --predict-output $parent_dir/data/AtlasZoomIn10/output/test/OnePrompt_points --csv-output-path $parent_dir/data/AtlasZoomIn10/output/test/OnePrompt_metrics.csv --image-size $img_resize $img_resize
    fi

    if [ "$upload_OnePrompt" = true ]; then
        echo "Checking OnePrompt output..."
        cd $parent_dir/code
        # python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_val_img --data-path $parent_dir/data/HAM10000/output/val/OnePrompt_points/ --image-size $img_resize $img_resize --image-num 1001
        # python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_test_img --data-path $parent_dir/data/HAM10000/output/test/OnePrompt_points/ --image-size $img_resize $img_resize --image-num 1002
        # python py_scripts/preprocessing/check_image_size.py --data-name ISIC2016_test_img --data-path $parent_dir/data/ISIC2016/output/test/OnePrompt_points/ --image-size $img_resize $img_resize --image-num 379
        # python py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_test_img --data-path $parent_dir/data/ISIC2017/output/test/OnePrompt_points/ --image-size $img_resize $img_resize --image-num 600
        # python py_scripts/preprocessing/check_image_size.py --data-name PH2_test_img --data-path $parent_dir/data/PH2/output/test/OnePrompt_points/ --image-size $img_resize $img_resize --image-num 200
        # python py_scripts/preprocessing/check_image_size.py --data-name Dermofit_test_img --data-path $parent_dir/data/Dermofit/output/test/OnePrompt_points/ --image-size $img_resize $img_resize --image-num 1300
        # python py_scripts/preprocessing/check_image_size.py --data-name Atlas_test_img --data-path $parent_dir/data/Atlas/output/test/OnePrompt_points/ --image-size $img_resize $img_resize --image-num 960
        python py_scripts/preprocessing/check_image_size.py --data-name AtlasZoomIn10_test_img --data-path $parent_dir/data/AtlasZoomIn10/output/test/OnePrompt_points/ --image-size $img_resize $img_resize --image-num 960
        echo "[√] OnePrompt output check pass."

        cd $parent_dir/code/compare_models/
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name OnePrompt-HAM10000_val --metrics-csv $parent_dir/data/HAM10000/output/val/OnePrompt_metrics.csv --timecost-txt $parent_dir/data/HAM10000/output/val/OnePrompt_points_timecost.txt --images-num 1001
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name OnePrompt-HAM10000_test --metrics-csv $parent_dir/data/HAM10000/output/test/OnePrompt_metrics.csv --timecost-txt $parent_dir/data/HAM10000/output/test/OnePrompt_points_timecost.txt --images-num 1002
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name OnePrompt-ISIC2016_test --metrics-csv $parent_dir/data/ISIC2016/output/test/OnePrompt_metrics.csv --timecost-txt $parent_dir/data/ISIC2016/output/test/OnePrompt_points_timecost.txt --images-num 379
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name OnePrompt-ISIC2017_test --metrics-csv $parent_dir/data/ISIC2017/output/test/OnePrompt_metrics.csv --timecost-txt $parent_dir/data/ISIC2017/output/test/OnePrompt_points_timecost.txt --images-num 600
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name OnePrompt-PH2_test --metrics-csv $parent_dir/data/PH2/output/test/OnePrompt_metrics.csv --timecost-txt $parent_dir/data/PH2/output/test/OnePrompt_points_timecost.txt --images-num 200
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name OnePrompt-Dermofit_test --metrics-csv $parent_dir/data/Dermofit/output/test/OnePrompt_metrics.csv --timecost-txt $parent_dir/data/Dermofit/output/test/OnePrompt_points_timecost.txt --images-num 1300
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name OnePrompt-Atlas_test --metrics-csv $parent_dir/data/Atlas/output/test/OnePrompt_metrics.csv --timecost-txt $parent_dir/data/Atlas/output/test/OnePrompt_points_timecost.txt --images-num 960
        python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name OnePrompt-AtlasZoomIn10_test --metrics-csv $parent_dir/data/AtlasZoomIn10/output/test/OnePrompt_metrics.csv --timecost-txt $parent_dir/data/AtlasZoomIn10/output/test/OnePrompt_points_timecost.txt --images-num 960
        echo "[√] OnePrompt Metrics upload finished."
    fi

    cd $parent_dir/code
    echo "✅ OnePrompt compare finished."
    echo "#########################################################################"
fi


############################
#  Scribble Saliency      ##
############################
if [ "$COMPARE_SCRIBBLESALIENCY" = true ]; then
    echo "Comparing with ScribbleSaliency..."
    mkdir -p $parent_dir/code/compare_models/reps/
    cd $parent_dir/code/compare_models/reps/
    if [ -d "ScribbleSaliency" ];
        then
            echo "[√] ScribbleSaliency already exists."
        else
            git clone https://github.com/JingZhang617/Scribble_Saliency.git ScribbleSaliency
            cd $parent_dir/code/compare_models/reps/ScribbleSaliency
    fi

    if [ -d "ScribbleSaliency_RCF" ];
        then
            echo "[√] ScribbleSaliency_RCF already exists."
        else
            git clone https://github.com/yun-liu/RCF-PyTorch.git ScribbleSaliency_RCF
            cd $parent_dir/code/compare_models/reps/ScribbleSaliency_RCF
    fi
    cd $parent_dir/code/compare_models/checkpoints
    if [ -d "ScribbleSaliency_RCF" ];
        then 
            echo "[√] ScribbleSaliency_RCF checkpoints exists."
        else
            mkdir ScribbleSaliency_RCF
            cd $parent_dir/code/compare_models/checkpoints/ScribbleSaliency_RCF
            wget -c -O "bsds500_pascal_model.pth" "https://example.com/placeholder/download"
    
    fi

    cd $parent_dir/code/compare_models/checkpoints
    if [ -d "ScribbleSaliency" ];
        then
            echo "[√] ScribbleSaliency checkpoints exists."
        else
            mkdir ScribbleSaliency
            cd $parent_dir/code/compare_models/checkpoints/ScribbleSaliency
            wget -c -O "vgg16-397923af.pth" "https://download.pytorch.org/models/vgg16-397923af.pth"
    fi

    if [ "$generate_dataset_ScribbleSaliency" = true ]; then
        echo "Preparing data for ScribbleSaliency..."
        cd $parent_dir/code/compare_models
        # HAM10000 train
        python ScribbleSalienc-prepare.py --positive-scribble-input $parent_dir/data/HAM10000/input/train/HAM10000_train_positive_lineScribble_42 --neagtive-scribble-input $parent_dir/data/HAM10000/input/train/HAM10000_train_negative_lineScribble_42 --mask-output $parent_dir/data/HAM10000/input/train/ScribbleSalienc/mask --gt-output $parent_dir/data/HAM10000/input/train/ScribbleSalienc/gt
        python ScribbleSalienc-convert_rgb2gray.py --img-input $parent_dir/data/HAM10000/input/train/HAM10000_img --gray-output $parent_dir/data/HAM10000/input/train/ScribbleSalienc/gray
        
        # # HAM10000 test
        # python ScribbleSalienc-prepare.py --positive-scribble-input $parent_dir/data/HAM10000/input/test/HAM10000_test_positive_lineScribble_42 --neagtive-scribble-input $parent_dir/data/HAM10000/input/test/HAM10000_test_negative_lineScribble_42 --mask-output $parent_dir/data/HAM10000/input/test/ScribbleSalienc/mask --gt-output $parent_dir/data/HAM10000/input/test/ScribbleSalienc/gt
        # python ScribbleSalienc-convert_rgb2gray.py --img-input $parent_dir/data/HAM10000/input/test/HAM10000_img --gray-output $parent_dir/data/HAM10000/input/test/ScribbleSalienc/gray
        
        # # HAM10000 val
        # python ScribbleSalienc-prepare.py --positive-scribble-input $parent_dir/data/HAM10000/input/val/HAM10000_val_positive_lineScribble_42 --neagtive-scribble-input $parent_dir/data/HAM10000/input/val/HAM10000_val_negative_lineScribble_42 --mask-output $parent_dir/data/HAM10000/input/val/ScribbleSalienc/mask --gt-output $parent_dir/data/HAM10000/input/val/ScribbleSalienc/gt
        # python ScribbleSalienc-convert_rgb2gray.py --img-input $parent_dir/data/HAM10000/input/val/HAM10000_img --gray-output $parent_dir/data/HAM10000/input/val/ScribbleSalienc/gray
        

        cd $parent_dir/code/compare_models/reps/ScribbleSaliency_RCF
        # HAM10000 train
        python test.py --checkpoint $parent_dir/code/compare_models/checkpoints/ScribbleSaliency_RCF/bsds500_pascal_model.pth --dataset $parent_dir/data/HAM10000/input/train/HAM10000_img --save-dir $parent_dir/data/HAM10000/input/train/ScribbleSalienc/edge
        
        # HAM10000 test
        # python test.py --checkpoint $parent_dir/code/compare_models/checkpoints/ScribbleSaliency_RCF/bsds500_pascal_model.pth --dataset $parent_dir/data/HAM10000/input/test/HAM10000_img --save-dir $parent_dir/data/HAM10000/input/test/ScribbleSalienc/edge
    
        # HAM10000 val
        # python test.py --checkpoint $parent_dir/code/compare_models/checkpoints/ScribbleSaliency_RCF/bsds500_pascal_model.pth --dataset $parent_dir/data/HAM10000/input/val/HAM10000_img --save-dir $parent_dir/data/HAM10000/input/val/ScribbleSalienc/edge
    
    fi

    if [ "$retrain_ScribbleSaliency" = true ]; then
        echo "Retraining ScribbleSaliency..."
        cd $parent_dir/code/compare_models/reps/ScribbleSaliency
        python train.py --image-root $parent_dir/data/HAM10000/input/train/HAM10000_img/ --gt-root $parent_dir/data/HAM10000/input/train/ScribbleSalienc/gt/ --mask_root $parent_dir/data/HAM10000/input/train/ScribbleSalienc/mask/ --edge_root $parent_dir/data/HAM10000/input/train/ScribbleSalienc/edge/ --gray_root $parent_dir/data/HAM10000/input/train/ScribbleSalienc/gray/
    fi

    if [ "$evaluate_ScribbleSaliency" = true ]; then
        cd $parent_dir/code/compare_models/reps/ScribbleSaliency

        # # HAM10000 val
        # rm -rf $parent_dir/data/HAM10000/output/val/ScribbleSaliency/
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id  python test.py --image-path $parent_dir/data/HAM10000/input/val/HAM10000_img/ --output-path $parent_dir/data/HAM10000/output/val/ScribbleSaliency/
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/HAM10000/output/val/ScribbleSaliency_timecost.txt


        # # HAM10000 test
        # rm -rf $parent_dir/data/HAM10000/output/test/ScribbleSaliency/
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id python test.py --image-path $parent_dir/data/HAM10000/input/test/HAM10000_img/ --output-path $parent_dir/data/HAM10000/output/test/ScribbleSaliency/
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/HAM10000/output/test/ScribbleSaliency_timecost.txt


        # # ISIC2016 test
        # rm -rf $parent_dir/data/ISIC2016/output/test/ScribbleSaliency/
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id python test.py --image-path $parent_dir/data/ISIC2016/input/test/ISIC2016_img/ --output-path $parent_dir/data/ISIC2016/output/test/ScribbleSaliency/
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/ISIC2016/output/test/ScribbleSaliency_timecost.txt

        # # ISIC2017 test
        # rm -rf $parent_dir/data/ISIC2017/output/test/ScribbleSaliency/
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id python test.py --image-path $parent_dir/data/ISIC2017/input/test/ISIC2017_img/ --output-path $parent_dir/data/ISIC2017/output/test/ScribbleSaliency/
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/ISIC2017/output/test/ScribbleSaliency_timecost.txt


        # # PH2 test
        # rm -rf $parent_dir/data/PH2/output/test/ScribbleSaliency/
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id python test.py --image-path $parent_dir/data/PH2/input/test/PH2_img/ --output-path $parent_dir/data/PH2/output/test/ScribbleSaliency/
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/PH2/output/test/ScribbleSaliency_timecost.txt


        # # Atlas test
        # rm -rf $parent_dir/data/Atlas/output/test/ScribbleSaliency/
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id python test.py --image-path $parent_dir/data/Atlas/input/test/Atlas_img/ --output-path $parent_dir/data/Atlas/output/test/ScribbleSaliency/
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/Atlas/output/test/ScribbleSaliency_timecost.txt


        # # Dermofit test
        # rm -rf $parent_dir/data/Dermofit/output/test/ScribbleSaliency/
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id python test.py --image-path $parent_dir/data/Dermofit/input/test/Dermofit_img/ --output-path $parent_dir/data/Dermofit/output/test/ScribbleSaliency/
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/Dermofit/output/test/ScribbleSaliency_timecost.txt

        # AtlasZoomIn10 test
        rm -rf $parent_dir/data/AtlasZoomIn10/output/test/ScribbleSaliency/
        start_time=$(date +%s)
        CUDA_VISIBLE_DEVICES=$gpu_id python test.py --image-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_img/ --output-path $parent_dir/data/AtlasZoomIn10/output/test/ScribbleSaliency/
        end_time=$(date +%s)
        cost_time=$((end_time - start_time))
        echo "Total cost [$cost_time] seconds" > $parent_dir/data/AtlasZoomIn10/output/test/ScribbleSaliency_timecost.txt

    fi

    if [ "$calculate_metrics_ScribbleSaliency" = true ]; then
        cd $parent_dir/code/compare_models/
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/HAM10000/input/val/HAM10000_seg --predict-output $parent_dir/data/HAM10000/output/val/ScribbleSaliency --csv-output-path $parent_dir/data/HAM10000/output/val/ScribbleSaliency_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/HAM10000/input/test/HAM10000_seg --predict-output $parent_dir/data/HAM10000/output/test/ScribbleSaliency --csv-output-path $parent_dir/data/HAM10000/output/test/ScribbleSaliency_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/ISIC2016/input/test/ISIC2016_seg --predict-output $parent_dir/data/ISIC2016/output/test/ScribbleSaliency --csv-output-path $parent_dir/data/ISIC2016/output/test/ScribbleSaliency_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/ISIC2017/input/test/ISIC2017_seg --predict-output $parent_dir/data/ISIC2017/output/test/ScribbleSaliency --csv-output-path $parent_dir/data/ISIC2017/output/test/ScribbleSaliency_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/PH2/input/test/PH2_seg --predict-output $parent_dir/data/PH2/output/test/ScribbleSaliency --csv-output-path $parent_dir/data/PH2/output/test/ScribbleSaliency_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/Dermofit/input/test/Dermofit_seg --predict-output $parent_dir/data/Dermofit/output/test/ScribbleSaliency --csv-output-path $parent_dir/data/Dermofit/output/test/ScribbleSaliency_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/Atlas/input/test/Atlas_seg --predict-output $parent_dir/data/Atlas/output/test/ScribbleSaliency --csv-output-path $parent_dir/data/Atlas/output/test/ScribbleSaliency_metrics.csv --image-size $img_resize $img_resize
        python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_seg --predict-output $parent_dir/data/AtlasZoomIn10/output/test/ScribbleSaliency --csv-output-path $parent_dir/data/AtlasZoomIn10/output/test/ScribbleSaliency_metrics.csv --image-size $img_resize $img_resize
    fi

    if [ "$upload_ScribbleSaliency" = true ]; then

        echo "Checking ScribbleSaliency output..."
        cd $parent_dir/code
        # python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_val_img --data-path $parent_dir/data/HAM10000/output/val/ScribbleSaliency/ --image-size $img_resize $img_resize --image-num 1001
        # python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_test_img --data-path $parent_dir/data/HAM10000/output/test/ScribbleSaliency/ --image-size $img_resize $img_resize --image-num 1002
        # python py_scripts/preprocessing/check_image_size.py --data-name ISIC2016_test_img --data-path $parent_dir/data/ISIC2016/output/test/ScribbleSaliency/ --image-size $img_resize $img_resize --image-num 379
        # python py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_test_img --data-path $parent_dir/data/ISIC2017/output/test/ScribbleSaliency/ --image-size $img_resize $img_resize --image-num 600
        # python py_scripts/preprocessing/check_image_size.py --data-name PH2_test_img --data-path $parent_dir/data/PH2/output/test/ScribbleSaliency/ --image-size $img_resize $img_resize --image-num 200
        # python py_scripts/preprocessing/check_image_size.py --data-name Dermofit_test_img --data-path $parent_dir/data/Dermofit/output/test/ScribbleSaliency/ --image-size $img_resize $img_resize --image-num 1300
        # python py_scripts/preprocessing/check_image_size.py --data-name Atlas_test_img --data-path $parent_dir/data/Atlas/output/test/ScribbleSaliency/ --image-size $img_resize $img_resize --image-num 960
        python py_scripts/preprocessing/check_image_size.py --data-name AtlasZoomIn10_test_img --data-path $parent_dir/data/AtlasZoomIn10/output/test/ScribbleSaliency/ --image-size $img_resize $img_resize --image-num 960
        echo "[√] nnUNet output check pass."


        cd $parent_dir/code/compare_models/
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name ScribbleSaliency-HAM10000_val --metrics-csv $parent_dir/data/HAM10000/output/val/ScribbleSaliency_metrics.csv --timecost-txt $parent_dir/data/HAM10000/output/val/ScribbleSaliency_timecost.txt --images-num 1001
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name ScribbleSaliency-HAM10000_test --metrics-csv $parent_dir/data/HAM10000/output/test/ScribbleSaliency_metrics.csv --timecost-txt $parent_dir/data/HAM10000/output/test/ScribbleSaliency_timecost.txt --images-num 1002
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name ScribbleSaliency-ISIC2016_test --metrics-csv $parent_dir/data/ISIC2016/output/test/ScribbleSaliency_metrics.csv --timecost-txt $parent_dir/data/ISIC2016/output/test/ScribbleSaliency_timecost.txt --images-num 379
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name ScribbleSaliency-ISIC2017_test --metrics-csv $parent_dir/data/ISIC2017/output/test/ScribbleSaliency_metrics.csv --timecost-txt $parent_dir/data/ISIC2017/output/test/ScribbleSaliency_timecost.txt --images-num 600
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name ScribbleSaliency-PH2_test --metrics-csv $parent_dir/data/PH2/output/test/ScribbleSaliency_metrics.csv --timecost-txt $parent_dir/data/PH2/output/test/ScribbleSaliency_timecost.txt --images-num 200
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name ScribbleSaliency-Dermofit_test --metrics-csv $parent_dir/data/Dermofit/output/test/ScribbleSaliency_metrics.csv --timecost-txt $parent_dir/data/Dermofit/output/test/ScribbleSaliency_timecost.txt --images-num 1300
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name ScribbleSaliency-Atlas_test --metrics-csv $parent_dir/data/Atlas/output/test/ScribbleSaliency_metrics.csv --timecost-txt $parent_dir/data/Atlas/output/test/ScribbleSaliency_timecost.txt --images-num 960
        python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name ScribbleSaliency-AtlasZoomIn10_test --metrics-csv $parent_dir/data/AtlasZoomIn10/output/test/ScribbleSaliency_metrics.csv --timecost-txt $parent_dir/data/AtlasZoomIn10/output/test/ScribbleSaliency_timecost.txt --images-num 960
        echo "[√] ScribbleSaliency Metrics upload finished."
    fi


    cd $parent_dir/code
    echo "✅ ScribbleSaliency compare finished."
    echo "#########################################################################"


fi
############################
#  ScribblePrompt         ##
############################
if [ "$COMPARE_SCRIBBLEPROMPT" = true ]; then
    echo "Comparing with ScribblePrompt..."
    mkdir -p $parent_dir/code/compare_models/reps/
    cd $parent_dir/code/compare_models/reps/
    if [ -d "ScribblePrompt" ];
    then
        echo "[√] ScribblePrompt already exists."
    else
        git clone https://github.com/halleewong/ScribblePrompt.git ScribblePrompt
        cd $parent_dir/code/compare_models/reps/ScribblePrompt
    fi

fi



############################
#  WSCOD                  ##
############################
if [ "$COMPARE_WSCOD" = true ]; then
    echo "Comparing with WSCOD..."
    mkdir -p $parent_dir/code/compare_models/reps/
    cd $parent_dir/code/compare_models/reps/
    if [ -d "WSCOD" ];
    then
        echo "[√] WSCOD already exists."
    else
        git clone https://github.com/dddraxxx/Weakly-Supervised-Camouflaged-Object-Detection-with-Scribble-Annotations.git WSCOD
        cd $parent_dir/code/compare_models/reps/WSCOD
        # pip install -e .
    fi

    if [ "$scribbles_generator_WSCOD" = true ]; then
        echo "Preparing data for WSCOD..."
        cd $parent_dir/code/compare_models/
        python WSCOD-scribbles_generator.py --seg-path $parent_dir/data/HAM10000/input/train/HAM10000_seg  --visable-out-path $parent_dir/data/HAM10000/input/train/HAM10000_scibble_visable  --WSCOD-out-path $parent_dir/data/HAM10000/input/train/HAM10000_scibble
        python WSCOD-scribbles_generator.py --seg-path $parent_dir/data/HAM10000/input/val/HAM10000_seg  --visable-out-path $parent_dir/data/HAM10000/input/val/HAM10000_scibble_visable  --WSCOD-out-path $parent_dir/data/HAM10000/input/val/HAM10000_scibble
        python WSCOD-scribbles_generator.py --seg-path $parent_dir/data/HAM10000/input/test/HAM10000_seg  --visable-out-path $parent_dir/data/HAM10000/input/test/HAM10000_scibble_visable  --WSCOD-out-path $parent_dir/data/HAM10000/input/test/HAM10000_scibble

        python WSCOD-scribbles_generator.py --seg-path $parent_dir/data/ISIC2016/input/train/ISIC2016_seg  --visable-out-path $parent_dir/data/ISIC2016/input/train/ISIC2016_scibble_visable  --WSCOD-out-path $parent_dir/data/ISIC2016/input/train/ISIC2016_scibble
        python WSCOD-scribbles_generator.py --seg-path $parent_dir/data/ISIC2016/input/test/ISIC2016_seg  --visable-out-path $parent_dir/data/ISIC2016/input/test/ISIC2016_scibble_visable  --WSCOD-out-path $parent_dir/data/ISIC2016/input/test/ISIC2016_scibble

        python WSCOD-scribbles_generator.py --seg-path $parent_dir/data/ISIC2017/input/train/ISIC2017_seg  --visable-out-path $parent_dir/data/ISIC2017/input/train/ISIC2017_scibble_visable  --WSCOD-out-path $parent_dir/data/ISIC2017/input/train/ISIC2017_scibble
        python WSCOD-scribbles_generator.py --seg-path $parent_dir/data/ISIC2017/input/val/ISIC2017_seg  --visable-out-path $parent_dir/data/ISIC2017/input/val/ISIC2017_scibble_visable  --WSCOD-out-path $parent_dir/data/ISIC2017/input/val/ISIC2017_scibble
        python WSCOD-scribbles_generator.py --seg-path $parent_dir/data/ISIC2017/input/test/ISIC2017_seg  --visable-out-path $parent_dir/data/ISIC2017/input/test/ISIC2017_scibble_visable  --WSCOD-out-path $parent_dir/data/ISIC2017/input/test/ISIC2017_scibble

        python WSCOD-scribbles_generator.py --seg-path $parent_dir/data/PH2/input/test/PH2_seg  --visable-out-path $parent_dir/data/PH2/input/test/PH2_scibble_visable  --WSCOD-out-path $parent_dir/data/PH2/input/test/PH2_scibble
        
        python WSCOD-scribbles_generator.py --seg-path $parent_dir/data/Atlas/input/test/Atlas_seg  --visable-out-path $parent_dir/data/Atlas/input/test/Atlas_scibble_visable  --WSCOD-out-path $parent_dir/data/Atlas/input/test/Atlas_scibble
        
        python WSCOD-scribbles_generator.py --seg-path $parent_dir/data/Dermofit/input/test/Dermofit_seg  --visable-out-path $parent_dir/data/Dermofit/input/test/Dermofit_scibble_visable  --WSCOD-out-path $parent_dir/data/Dermofit/input/test/Dermofit_scibble

    fi
    if [ "$prepare_WSCOD" = true ]; then
        echo "Prepare WSCOD..."
        cd $parent_dir/code/compare_models/
        python WSCOD-generate_record.py --input $parent_dir/data/HAM10000/input/train/HAM10000_img --output $parent_dir/data/HAM10000/input/train.txt
        python WSCOD-generate_record.py --input $parent_dir/data/HAM10000/input/val/HAM10000_img --output $parent_dir/data/HAM10000/input/val.txt
        python WSCOD-generate_record.py --input $parent_dir/data/HAM10000/input/test/HAM10000_img --output $parent_dir/data/HAM10000/input/test.txt

        cd $parent_dir/code/compare_models/reps/WSCOD
    fi

    if [ "$retrain_WSCOD" = true ]; then
        echo "Retrain WSCOD..."

        cd $parent_dir/code/compare_models/reps/WSCOD
        rm -rf $parent_dir/code/compare_models/checkpoints/WSCOD
        if [ -d "assets" ];
        then
            echo "[√] WSCOD assets already exists."
        else
            mkdir -p assets
            cd assets
            wget -O "resnet50-19c8e357.pth" "https://example.com/placeholder/download"
        fi

        cd $parent_dir/code/compare_models/reps/WSCOD
        CUDA_VISIBLE_DEVICES=$gpu_id python train.py --root $parent_dir/data/HAM10000/input --output $parent_dir/code/compare_models/checkpoints/WSCOD

    fi

    if [ "$evaluate_WSCOD" = true ]; then
        echo "Evaluate WSCOD..."

        cd $parent_dir/code/compare_models/reps/WSCOD

        # HAM10000 test
        start_time=$(date +%s)
        CUDA_VISIBLE_DEVICES=$gpu_id python test.py --dataset $parent_dir/data/HAM10000/input/test/ --checkpoint $parent_dir/code/compare_models/checkpoints/WSCOD/model-50
        end_time=$(date +%s)
        cost_time=$((end_time - start_time))
        echo "Total cost [$cost_time] seconds" > $parent_dir/data/HAM10000/output/test/WSCOD_timecost.txt

    fi

fi



# ############################
# #  SAM Specific Points    ##
# ############################
if [ "$COMPARE_SAMSPECIFIC" = true ]; then
    echo "Comparing with SAM specific points..."
    if [ "$evaluate_SAMSpecific" = true ]; then
        echo "Evaluating on SAMSpecific..."
        cd $parent_dir/code/
        # python ./compare_models/SAMSpecific_predict.py --input-path $parent_dir/data/HAM10000/input/val/HAM10000_img --gt-path $parent_dir/data/HAM10000/input/val/HAM10000_seg --output-path $parent_dir/data/HAM10000/output/val/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth --point-csv-path $parent_dir/data/HAM10000/input/HAM10000_val_prompts_$labeller.csv
        # python ./compare_models/SAMSpecific_predict.py --input-path $parent_dir/data/HAM10000/input/test/HAM10000_img --gt-path $parent_dir/data/HAM10000/input/test/HAM10000_seg --output-path $parent_dir/data/HAM10000/output/test/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth --point-csv-path $parent_dir/data/HAM10000/input/HAM10000_test_prompts_$labeller.csv
        # python ./compare_models/SAMSpecific_predict.py --input-path $parent_dir/data/ISIC2016/input/test/ISIC2016_img --gt-path $parent_dir/data/ISIC2016/input/test/ISIC2016_seg --output-path $parent_dir/data/ISIC2016/output/test/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth --point-csv-path $parent_dir/data/ISIC2016/input/ISIC2016_test_prompts_$labeller.csv
        # python ./compare_models/SAMSpecific_predict.py --input-path $parent_dir/data/ISIC2017/input/test/ISIC2017_img --gt-path $parent_dir/data/ISIC2017/input/test/ISIC2017_seg --output-path $parent_dir/data/ISIC2017/output/test/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth --point-csv-path $parent_dir/data/ISIC2017/input/ISIC2017_test_prompts_$labeller.csv
        # python ./compare_models/SAMSpecific_predict.py --input-path $parent_dir/data/PH2/input/test/PH2_img --gt-path $parent_dir/data/PH2/input/test/PH2_seg --output-path $parent_dir/data/PH2/output/test/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth --point-csv-path $parent_dir/data/PH2/input/PH2_test_prompts_$labeller.csv
        # python ./compare_models/SAMSpecific_predict.py --input-path $parent_dir/data/Dermofit/input/test/Dermofit_img --gt-path $parent_dir/data/Dermofit/input/test/Dermofit_seg --output-path $parent_dir/data/Dermofit/output/test/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth --point-csv-path $parent_dir/data/Dermofit/input/Dermofit_test_prompts_$labeller.csv
        # python ./compare_models/SAMSpecific_predict.py --input-path $parent_dir/data/Atlas/input/test/Atlas_img --gt-path $parent_dir/data/Atlas/input/test/Atlas_seg --output-path $parent_dir/data/Atlas/output/test/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth --point-csv-path $parent_dir/data/Atlas/input/Atlas_test_prompts_$labeller.csv
        python ./compare_models/SAMSpecific_predict.py --input-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_img --gt-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_seg --output-path $parent_dir/data/AtlasZoomIn10/output/test/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth --point-csv-path $parent_dir/data/AtlasZoomIn10/input/AtlasZoomIn10_test_prompts_$labeller.csv
        python ./compare_models/SAMSpecific_predict.py --input-path $parent_dir/data/AtlasZoomIn35/input/test/AtlasZoomIn35_img --gt-path $parent_dir/data/AtlasZoomIn35/input/test/AtlasZoomIn35_seg --output-path $parent_dir/data/AtlasZoomIn35/output/test/ --checkpoint $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth --point-csv-path $parent_dir/data/AtlasZoomIn35/input/AtlasZoomIn35_test_prompts_$labeller.csv
    fi

    if [ "$calculate_metrics_SAMSpecific" = true ]; then
        echo "Calculate SAMAuto_timecost from csv..."
        cd $parent_dir/code/py_scripts/final
        # python calculate_timecost.py --input-csv-path $parent_dir/data/HAM10000/output/val/SAMSpecific_timecost.csv --output-txt-path $parent_dir/data/HAM10000/output/val/SAMSpecific_timecost.txt --images-num 1001
        # python calculate_timecost.py --input-csv-path $parent_dir/data/HAM10000/output/test/SAMSpecific_timecost.csv --output-txt-path $parent_dir/data/HAM10000/output/test/SAMSpecific_timecost.txt --images-num 1002
        # python calculate_timecost.py --input-csv-path $parent_dir/data/ISIC2016/output/test/SAMSpecific_timecost.csv --output-txt-path $parent_dir/data/ISIC2016/output/test/SAMSpecific_timecost.txt --images-num 379
        # python calculate_timecost.py --input-csv-path $parent_dir/data/ISIC2017/output/test/SAMSpecific_timecost.csv --output-txt-path $parent_dir/data/ISIC2017/output/test/SAMSpecific_timecost.txt --images-num 600
        # python calculate_timecost.py --input-csv-path $parent_dir/data/PH2/output/test/SAMSpecific_timecost.csv --output-txt-path $parent_dir/data/PH2/output/test/SAMSpecific_timecost.txt --images-num 200
        # python calculate_timecost.py --input-csv-path $parent_dir/data/Dermofit/output/test/SAMSpecific_timecost.csv --output-txt-path $parent_dir/data/Dermofit/output/test/SAMSpecific_timecost.txt --images-num 1300
        # python calculate_timecost.py --input-csv-path $parent_dir/data/Atlas/output/test/SAMSpecific_timecost.csv --output-txt-path $parent_dir/data/Atlas/output/test/SAMSpecific_timecost.txt --images-num 960
        python calculate_timecost.py --input-csv-path $parent_dir/data/AtlasZoomIn10/output/test/SAMSpecific_timecost.csv --output-txt-path $parent_dir/data/AtlasZoomIn10/output/test/SAMSpecific_timecost.txt --images-num 960
        python calculate_timecost.py --input-csv-path $parent_dir/data/AtlasZoomIn35/output/test/SAMSpecific_timecost.csv --output-txt-path $parent_dir/data/AtlasZoomIn35/output/test/SAMSpecific_timecost.txt --images-num 960
        echo "[√] SAMSpecific_timecost calculation finish."

        cd $parent_dir/code/compare_models/
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/HAM10000/input/val/HAM10000_seg --predict-output $parent_dir/data/HAM10000/output/val/SAMSpecific --csv-output-path $parent_dir/data/HAM10000/output/val/SAMSpecific_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/HAM10000/input/test/HAM10000_seg --predict-output $parent_dir/data/HAM10000/output/test/SAMSpecific --csv-output-path $parent_dir/data/HAM10000/output/test/SAMSpecific_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/ISIC2016/input/test/ISIC2016_seg --predict-output $parent_dir/data/ISIC2016/output/test/SAMSpecific --csv-output-path $parent_dir/data/ISIC2016/output/test/SAMSpecific_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/ISIC2017/input/test/ISIC2017_seg --predict-output $parent_dir/data/ISIC2017/output/test/SAMSpecific --csv-output-path $parent_dir/data/ISIC2017/output/test/SAMSpecific_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/PH2/input/test/PH2_seg --predict-output $parent_dir/data/PH2/output/test/SAMSpecific --csv-output-path $parent_dir/data/PH2/output/test/SAMSpecific_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/Dermofit/input/test/Dermofit_seg --predict-output $parent_dir/data/Dermofit/output/test/SAMSpecific --csv-output-path $parent_dir/data/Dermofit/output/test/SAMSpecific_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/Atlas/input/test/Atlas_seg --predict-output $parent_dir/data/Atlas/output/test/SAMSpecific --csv-output-path $parent_dir/data/Atlas/output/test/SAMSpecific_metrics.csv --image-size $img_resize $img_resize
        python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_seg --predict-output $parent_dir/data/AtlasZoomIn10/output/test/SAMSpecific --csv-output-path $parent_dir/data/AtlasZoomIn10/output/test/SAMSpecific_metrics.csv --image-size $img_resize $img_resize
        python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/AtlasZoomIn35/input/test/AtlasZoomIn35_seg --predict-output $parent_dir/data/AtlasZoomIn35/output/test/SAMSpecific --csv-output-path $parent_dir/data/AtlasZoomIn35/output/test/SAMSpecific_metrics.csv --image-size $img_resize $img_resize
        echo "[√] SAMSpecific Metrics calculation finish."
    fi

    if [ "$upload_SAMSpecific" = true ]; then
        echo "Checking SAMSpecific output..."
        cd $parent_dir/code
        # python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_val_img --data-path $parent_dir/data/HAM10000/output/val/SAMSpecific/ --image-size $img_resize $img_resize --image-num 1001
        # python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_test_img --data-path $parent_dir/data/HAM10000/output/test/SAMSpecific/ --image-size $img_resize $img_resize --image-num 1002
        # python py_scripts/preprocessing/check_image_size.py --data-name ISIC2016_test_img --data-path $parent_dir/data/ISIC2016/output/test/SAMSpecific/ --image-size $img_resize $img_resize --image-num 379
        # python py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_test_img --data-path $parent_dir/data/ISIC2017/output/test/SAMSpecific/ --image-size $img_resize $img_resize --image-num 600
        # python py_scripts/preprocessing/check_image_size.py --data-name PH2_test_img --data-path $parent_dir/data/PH2/output/test/SAMSpecific/ --image-size $img_resize $img_resize --image-num 200
        # python py_scripts/preprocessing/check_image_size.py --data-name Dermofit_test_img --data-path $parent_dir/data/Dermofit/output/test/SAMSpecific/ --image-size $img_resize $img_resize --image-num 1300
        # python py_scripts/preprocessing/check_image_size.py --data-name Atlas_test_img --data-path $parent_dir/data/Atlas/output/test/SAMSpecific/ --image-size $img_resize $img_resize --image-num 960
        python py_scripts/preprocessing/check_image_size.py --data-name AtlasZoomIn10_test_img --data-path $parent_dir/data/AtlasZoomIn10/output/test/SAMSpecific/ --image-size $img_resize $img_resize --image-num 960
        python py_scripts/preprocessing/check_image_size.py --data-name AtlasZoomIn35_test_img --data-path $parent_dir/data/AtlasZoomIn35/output/test/SAMSpecific/ --image-size $img_resize $img_resize --image-num 960
        echo "[√] SAMSpecific output check pass."

        cd $parent_dir/code/compare_models/
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMSpecific-HAM10000_val --metrics-csv $parent_dir/data/HAM10000/output/val/SAMSpecific_metrics.csv --timecost-txt $parent_dir/data/HAM10000/output/val/SAMSpecific_timecost.txt --images-num 1001
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMSpecific-HAM10000_test --metrics-csv $parent_dir/data/HAM10000/output/test/SAMSpecific_metrics.csv --timecost-txt $parent_dir/data/HAM10000/output/test/SAMSpecific_timecost.txt --images-num 1002
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMSpecific-ISIC2016_test --metrics-csv $parent_dir/data/ISIC2016/output/test/SAMSpecific_metrics.csv --timecost-txt $parent_dir/data/ISIC2016/output/test/SAMSpecific_timecost.txt --images-num 379
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMSpecific-ISIC2017_test --metrics-csv $parent_dir/data/ISIC2017/output/test/SAMSpecific_metrics.csv --timecost-txt $parent_dir/data/ISIC2017/output/test/SAMSpecific_timecost.txt --images-num 600
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMSpecific-PH2_test --metrics-csv $parent_dir/data/PH2/output/test/SAMSpecific_metrics.csv --timecost-txt $parent_dir/data/PH2/output/test/SAMSpecific_timecost.txt --images-num 200
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMSpecific-Dermofit_test --metrics-csv $parent_dir/data/Dermofit/output/test/SAMSpecific_metrics.csv --timecost-txt $parent_dir/data/Dermofit/output/test/SAMSpecific_timecost.txt --images-num 1300
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMSpecific-Atlas_test --metrics-csv $parent_dir/data/Atlas/output/test/SAMSpecific_metrics.csv --timecost-txt $parent_dir/data/Atlas/output/test/SAMSpecific_timecost.txt --images-num 960
        python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMSpecific-AtlasZoomIn10_test --metrics-csv $parent_dir/data/AtlasZoomIn10/output/test/SAMSpecific_metrics.csv --timecost-txt $parent_dir/data/AtlasZoomIn10/output/test/SAMSpecific_timecost.txt --images-num 960
        python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name SAMSpecific-AtlasZoomIn35_test --metrics-csv $parent_dir/data/AtlasZoomIn35/output/test/SAMSpecific_metrics.csv --timecost-txt $parent_dir/data/AtlasZoomIn35/output/test/SAMSpecific_timecost.txt --images-num 960
        echo "[√] SAMSpecific Metrics upload finished."
    fi
fi


# ############################
# #  MedicalSAMAdatpter     ##
# ############################
# if [ "$COMPARE_MEDICALSAM" = true ]; then
#     echo "Comparing with MedicalSAM..."
#     mkdir -p $parent_dir/code/compare_models/reps/
#     cd $parent_dir/code/compare_models/reps/
#     if [ -d "MedicalSAM" ];
#     then
#         echo "[√] MedicalSAM already exists."
#     else
#         git clone https://github.com/MedicineToken/Medical-SAM-Adapter.git MedicalSAM
#         cd $parent_dir/code/compare_models/reps/MedicalSAM
#         # pip install -e .
#     fi
#     cp $parent_dir/code/compare_models/MedicalSAM-dataset.py $parent_dir/code/compare_models/reps/MedicalSAM/dataset/dataset.py
#     cp $parent_dir/code/compare_models/MedicalSAM-__init__.py $parent_dir/code/compare_models/reps/MedicalSAM/dataset/__init__.py
#     cp $parent_dir/code/compare_models/MedicalSAM-cfg.py $parent_dir/code/compare_models/reps/MedicalSAM/cfg.py
        
#     if [ "$retrain_MedicalSAM" = true ]; then
#         # cd $parent_dir/code/py_scripts/preprocessing/

#         # # HAM10000
#         # python generate_monai_json.py --dataset-name HAM10000 --path $parent_dir/data/HAM10000/input
#         # python generate_dataset_csv.py --img-path $parent_dir/data/HAM10000/input/train/HAM10000_img --seg-path $parent_dir/data/HAM10000/input/train/HAM10000_seg --csv-path $parent_dir/data/HAM10000/input/HAM10000_train.csv
#         # python generate_dataset_csv.py --img-path $parent_dir/data/HAM10000/input/val/HAM10000_img --seg-path $parent_dir/data/HAM10000/input/val/HAM10000_seg --csv-path $parent_dir/data/HAM10000/input/HAM10000_val.csv
#         # python generate_dataset_csv.py --img-path $parent_dir/data/HAM10000/input/test/HAM10000_img --seg-path $parent_dir/data/HAM10000/input/test/HAM10000_seg --csv-path $parent_dir/data/HAM10000/input/HAM10000_test.csv
        
#         # # Other test dataset
#         # python generate_dataset_csv.py --img-path $parent_dir/data/ISIC2016/input/test/ISIC2016_img --seg-path $parent_dir/data/ISIC2016/input/test/ISIC2016_seg --csv-path $parent_dir/data/ISIC2016/input/ISIC2016_test.csv
#         # python generate_dataset_csv.py --img-path $parent_dir/data/ISIC2017/input/test/ISIC2017_img --seg-path $parent_dir/data/ISIC2017/input/test/ISIC2017_seg --csv-path $parent_dir/data/ISIC2017/input/ISIC2017_test.csv
#         # python generate_dataset_csv.py --img-path $parent_dir/data/PH2/input/test/PH2_img --seg-path $parent_dir/data/PH2/input/test/PH2_seg --csv-path $parent_dir/data/PH2/input/PH2_test.csv
#         # python generate_dataset_csv.py --img-path $parent_dir/data/Atlas/input/test/Atlas_img --seg-path $parent_dir/data/Atlas/input/test/Atlas_seg --csv-path $parent_dir/data/Atlas/input/Atlas_test.csv
#         # python generate_dataset_csv.py --img-path $parent_dir/data/Dermofit/input/test/Dermofit_img --seg-path $parent_dir/data/Dermofit/input/test/Dermofit_seg --csv-path $parent_dir/data/Dermofit/input/Dermofit_test.csv

#         # cp $parent_dir/code/compare_models/MedicalSAM-train.py $parent_dir/code/compare_models/reps/MedicalSAM/train.py
#         cd $parent_dir/code/compare_models/reps/MedicalSAM
#         CUDA_VISIBLE_DEVICES=$gpu_id python train.py -net sam -mod sam_adpt -exp_name HAM10000 -sam_ckpt $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth -image_size 1024 -b 2 -dataset HAM10000 -data_path $parent_dir/data/HAM10000/input -val_freq 1 -vis 1

#     fi

#     if [ "$evaluate_MedicalSAM" = true ]; then
#         # 需要手动复制pth文件到save路径下
#         cd $parent_dir/code/compare_models/reps/MedicalSAM
#         # weight_path=$parent_dir/code/compare_models/reps/MedicalSAM/logs/save/checkpoint_best_epoch2.pth
#         # weight_path=$parent_dir/code/compare_models/reps/MedicalSAM/logs/save/checkpoint_best_auto.pth
#         weight_path=$parent_dir/code/compare_models/reps/MedicalSAM/logs/save/Melanoma_Photo_SAM_1024.pth

#         echo "Evaluating on HAM10000 test..."
#         rm -rf $parent_dir/data/HAM10000/output/test/MedicalSAM
#         start_time=$(date +%s)
#         CUDA_VISIBLE_DEVICES=$gpu_id python val.py -net sam -mod sam_adpt -exp_name HAM10000_test -sam_ckpt $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth -image_size 1024 -b 2 -dataset HAM10000_test -data_path $parent_dir/data/HAM10000/input -val_freq 1 -vis 1 -output_path $parent_dir/data/HAM10000/output/test/MedicalSAM -weights $weight_path
#         end_time=$(date +%s)
#         cost_time=$((end_time - start_time))
#         echo "Total cost [$cost_time] seconds" > $parent_dir/data/HAM10000/output/test/MedicalSAM_timecost.txt

#         echo "Evaluating on HAM10000 val..."
#         rm -rf $parent_dir/data/HAM10000/output/val/MedicalSAM
#         start_time=$(date +%s)
#         CUDA_VISIBLE_DEVICES=$gpu_id python val.py -net sam -mod sam_adpt -exp_name HAM10000_val -sam_ckpt $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth -image_size 1024 -b 2 -dataset HAM10000_val -data_path $parent_dir/data/HAM10000/input -val_freq 1 -vis 1 -output_path $parent_dir/data/HAM10000/output/val/MedicalSAM -weights $weight_path
#         end_time=$(date +%s)
#         cost_time=$((end_time - start_time))
#         echo "Total cost [$cost_time] seconds" > $parent_dir/data/HAM10000/output/val/MedicalSAM_timecost.txt

#         echo "Evaluating on ISIC2016 test..."
#         rm -rf $parent_dir/data/ISIC2016/output/test/MedicalSAM
#         start_time=$(date +%s)
#         CUDA_VISIBLE_DEVICES=$gpu_id python val.py -net sam -mod sam_adpt -exp_name ISIC2016_test -sam_ckpt $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth -image_size 1024 -b 2 -dataset ISIC2016_test -data_path $parent_dir/data/ISIC2016/input -val_freq 1 -vis 1 -output_path $parent_dir/data/ISIC2016/output/test/MedicalSAM -weights $weight_path
#         end_time=$(date +%s)
#         cost_time=$((end_time - start_time))
#         echo "Total cost [$cost_time] seconds" > $parent_dir/data/ISIC2016/output/test/MedicalSAM_timecost.txt

#         echo "Evaluating on ISIC2017 test..."
#         rm -rf $parent_dir/data/ISIC2017/output/test/MedicalSAM
#         start_time=$(date +%s)
#         CUDA_VISIBLE_DEVICES=$gpu_id python val.py -net sam -mod sam_adpt -exp_name ISIC2017_test -sam_ckpt $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth -image_size 1024 -b 2 -dataset ISIC2017_test -data_path $parent_dir/data/ISIC2017/input -val_freq 1 -vis 1 -output_path $parent_dir/data/ISIC2017/output/test/MedicalSAM -weights $weight_path
#         end_time=$(date +%s)
#         cost_time=$((end_time - start_time))
#         echo "Total cost [$cost_time] seconds" > $parent_dir/data/ISIC2017/output/test/MedicalSAM_timecost.txt

#         echo "Evaluating on PH2 test..."
#         rm -rf $parent_dir/data/PH2/output/test/MedicalSAM
#         start_time=$(date +%s)
#         CUDA_VISIBLE_DEVICES=$gpu_id python val.py -net sam -mod sam_adpt -exp_name PH2_test -sam_ckpt $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth -image_size 1024 -b 2 -dataset PH2_test -data_path $parent_dir/data/PH2/input -val_freq 1 -vis 1 -output_path $parent_dir/data/PH2/output/test/MedicalSAM -weights $weight_path
#         end_time=$(date +%s)
#         cost_time=$((end_time - start_time))
#         echo "Total cost [$cost_time] seconds" > $parent_dir/data/PH2/output/test/MedicalSAM_timecost.txt

#         echo "Evaluating on Dermofit test..."
#         rm -rf $parent_dir/data/Dermofit/output/test/MedicalSAM
#         start_time=$(date +%s)
#         CUDA_VISIBLE_DEVICES=$gpu_id python val.py -net sam -mod sam_adpt -exp_name Dermofit_test -sam_ckpt $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth -image_size 1024 -b 2 -dataset Dermofit_test -data_path $parent_dir/data/Dermofit/input -val_freq 1 -vis 1 -output_path $parent_dir/data/Dermofit/output/test/MedicalSAM -weights $weight_path
#         end_time=$(date +%s)
#         cost_time=$((end_time - start_time))
#         echo "Total cost [$cost_time] seconds" > $parent_dir/data/Dermofit/output/test/MedicalSAM_timecost.txt
    
#         # cd $parent_dir/code/py_scripts/preprocessing/
#         # python generate_dataset_csv.py --img-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_img --seg-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_seg --csv-path $parent_dir/data/AtlasZoomIn10/input/AtlasZoomIn10_test.csv

#         echo "Evaluating on AtlasZoomIn10 test..."
#         rm -rf $parent_dir/data/AtlasZoomIn10/output/test/MedicalSAM
#         start_time=$(date +%s)
#         CUDA_VISIBLE_DEVICES=$gpu_id python val.py -net sam -mod sam_adpt -exp_name AtlasZoomIn10_test -sam_ckpt $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth -image_size 1024 -b 2 -dataset AtlasZoomIn10_test -data_path $parent_dir/data/AtlasZoomIn10/input -val_freq 1 -vis 1 -output_path $parent_dir/data/AtlasZoomIn10/output/test/MedicalSAM -weights $weight_path
#         end_time=$(date +%s)
#         cost_time=$((end_time - start_time))
#         echo "Total cost [$cost_time] seconds" > $parent_dir/data/AtlasZoomIn10/output/test/MedicalSAM_timecost.txt
    
#     fi

#     if [ "$calculate_metrics_MedicalSAM" = true ]; then
#         cd $parent_dir/code/compare_models/
#         python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/HAM10000/input/val/HAM10000_seg --predict-output $parent_dir/data/HAM10000/output/val/MedicalSAM --csv-output-path $parent_dir/data/HAM10000/output/val/MedicalSAM_metrics.csv --image-size $img_resize $img_resize
#         python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/HAM10000/input/test/HAM10000_seg --predict-output $parent_dir/data/HAM10000/output/test/MedicalSAM --csv-output-path $parent_dir/data/HAM10000/output/test/MedicalSAM_metrics.csv --image-size $img_resize $img_resize
#         python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/ISIC2016/input/test/ISIC2016_seg --predict-output $parent_dir/data/ISIC2016/output/test/MedicalSAM --csv-output-path $parent_dir/data/ISIC2016/output/test/MedicalSAM_metrics.csv --image-size $img_resize $img_resize
#         python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/ISIC2017/input/test/ISIC2017_seg --predict-output $parent_dir/data/ISIC2017/output/test/MedicalSAM --csv-output-path $parent_dir/data/ISIC2017/output/test/MedicalSAM_metrics.csv --image-size $img_resize $img_resize
#         python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/PH2/input/test/PH2_seg --predict-output $parent_dir/data/PH2/output/test/MedicalSAM --csv-output-path $parent_dir/data/PH2/output/test/MedicalSAM_metrics.csv --image-size $img_resize $img_resize
#         python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/Dermofit/input/test/Dermofit_seg --predict-output $parent_dir/data/Dermofit/output/test/MedicalSAM --csv-output-path $parent_dir/data/Dermofit/output/test/MedicalSAM_metrics.csv --image-size $img_resize $img_resize
#         # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/Atlas/input/test/Atlas_seg --predict-output $parent_dir/data/Atlas/output/test/MedicalSAM --csv-output-path $parent_dir/data/Atlas/output/test/MedicalSAM_metrics.csv --image-size $img_resize $img_resize
#         python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_seg --predict-output $parent_dir/data/AtlasZoomIn10/output/test/MedicalSAM --csv-output-path $parent_dir/data/AtlasZoomIn10/output/test/MedicalSAM_metrics.csv --image-size $img_resize $img_resize
#     fi

#     if [ "$upload_MedicalSAM" = true ]; then
#         echo "Checking MedicalSAM output..."
#         cd $parent_dir/code
#         python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_val_img --data-path $parent_dir/data/HAM10000/output/val/MedicalSAM/ --image-size $img_resize $img_resize --image-num 1001
#         python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_test_img --data-path $parent_dir/data/HAM10000/output/test/MedicalSAM/ --image-size $img_resize $img_resize --image-num 1002
#         python py_scripts/preprocessing/check_image_size.py --data-name ISIC2016_test_img --data-path $parent_dir/data/ISIC2016/output/test/MedicalSAM/ --image-size $img_resize $img_resize --image-num 379
#         python py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_test_img --data-path $parent_dir/data/ISIC2017/output/test/MedicalSAM/ --image-size $img_resize $img_resize --image-num 600
#         python py_scripts/preprocessing/check_image_size.py --data-name PH2_test_img --data-path $parent_dir/data/PH2/output/test/MedicalSAM/ --image-size $img_resize $img_resize --image-num 200
#         python py_scripts/preprocessing/check_image_size.py --data-name Dermofit_test_img --data-path $parent_dir/data/Dermofit/output/test/MedicalSAM/ --image-size $img_resize $img_resize --image-num 1300
#         # python py_scripts/preprocessing/check_image_size.py --data-name Atlas_test_img --data-path $parent_dir/data/Atlas/output/test/MedicalSAM/ --image-size $img_resize $img_resize --image-num 960
#         python py_scripts/preprocessing/check_image_size.py --data-name AtlasZoomIn10_test_img --data-path $parent_dir/data/AtlasZoomIn10/output/test/MedicalSAM/ --image-size $img_resize $img_resize --image-num 960
#         echo "[√] MedicalSAM output check pass."

#         cd $parent_dir/code/compare_models/
#         python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name MedicalSAM-HAM10000_val --metrics-csv $parent_dir/data/HAM10000/output/val/MedicalSAM_metrics.csv --timecost-txt $parent_dir/data/HAM10000/output/val/MedicalSAM_timecost.txt --images-num 1001
#         python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name MedicalSAM-HAM10000_test --metrics-csv $parent_dir/data/HAM10000/output/test/MedicalSAM_metrics.csv --timecost-txt $parent_dir/data/HAM10000/output/test/MedicalSAM_timecost.txt --images-num 1002
#         python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name MedicalSAM-ISIC2016_test --metrics-csv $parent_dir/data/ISIC2016/output/test/MedicalSAM_metrics.csv --timecost-txt $parent_dir/data/ISIC2016/output/test/MedicalSAM_timecost.txt --images-num 379
#         python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name MedicalSAM-ISIC2017_test --metrics-csv $parent_dir/data/ISIC2017/output/test/MedicalSAM_metrics.csv --timecost-txt $parent_dir/data/ISIC2017/output/test/MedicalSAM_timecost.txt --images-num 600
#         python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name MedicalSAM-PH2_test --metrics-csv $parent_dir/data/PH2/output/test/MedicalSAM_metrics.csv --timecost-txt $parent_dir/data/PH2/output/test/MedicalSAM_timecost.txt --images-num 200
#         python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name MedicalSAM-Dermofit_test --metrics-csv $parent_dir/data/Dermofit/output/test/MedicalSAM_metrics.csv --timecost-txt $parent_dir/data/Dermofit/output/test/MedicalSAM_timecost.txt --images-num 1300
#         # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name MedicalSAM-Atlas_test --metrics-csv $parent_dir/data/Atlas/output/test/MedicalSAM_metrics.csv --timecost-txt $parent_dir/data/Atlas/output/test/MedicalSAM_timecost.txt --images-num 960
#         python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name MedicalSAM-AtlasZoomIn10_test --metrics-csv $parent_dir/data/AtlasZoomIn10/output/test/MedicalSAM_metrics.csv --timecost-txt $parent_dir/data/AtlasZoomIn10/output/test/MedicalSAM_timecost.txt --images-num 960
#         echo "[√] MedicalSAM Metrics upload finished."
#     fi

#     cd $parent_dir/code
#     echo "✅ MedicalSAM compare finished."
#     echo "#########################################################################"
# fi


############################
#  MedicalSAMAuto              ##
############################
if [ "$COMPARE_MedSAMAuto" = true ]; then
    echo "Comparing with MedicalSAM Auto..."
    mkdir -p $parent_dir/code/compare_models/reps/
    cd $parent_dir/code/compare_models/reps/
    if [ -d "MedicalSAM" ];
    then
        echo "[√] MedicalSAM already exists."
    else
        git clone https://github.com/bowang-lab/MedSAM.git MedicalSAM
        cd $parent_dir/code/compare_models/reps/MedicalSAM
        # pip install -e .
    fi

    # 下载 https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN
    # 到 work_dir/MedSAM/medsam_vit_b

    # cd $parent_dir/code/compare_models/reps/MedicalSAM/work_dir/MedSAM/medsam_vit_b
    # if [ -d "MedicalSAM" ];then
    #     echo "[√] MedicalSAM checkpoint exists."
    # else
    #     wget https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN?usp=drive_link
    #     cd $parent_dir/code/compare_models/reps/MedicalSAM
    # fi

    if [ "$evaluate_MedSAMAuto" = true ]; then
        # 需要手动复制pth文件到save路径下
        cd $parent_dir/code/compare_models/reps/MedicalSAM

        # echo "Evaluating on HAM10000 test..."
        # output_path=$parent_dir/data/HAM10000/output/test/MedSAMAuto
        # input_path=$parent_dir/data/HAM10000/input/test/HAM10000_img
        # seg_path=$parent_dir/data/HAM10000/input/test/HAM10000_seg
        # point_path=$parent_dir/data/HAM10000/input/HAM10000_test_prompts_42.csv
        # rm -rf $output_path
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id python MedSAMAuto_Inference.py -i $input_path -o $output_path -gt $seg_path
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/HAM10000/output/test/MedSAMAuto_timecost.txt

        # echo "Evaluating on HAM10000 val..."
        # output_path=$parent_dir/data/HAM10000/output/val/MedSAMAuto
        # input_path=$parent_dir/data/HAM10000/input/val/HAM10000_img
        # seg_path=$parent_dir/data/HAM10000/input/val/HAM10000_seg
        # point_path=$parent_dir/data/HAM10000/input/HAM10000_val_prompts_42.csv
        # rm -rf $output_path
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id python MedSAMAuto_Inference.py -i $input_path -o $output_path -gt $seg_path
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/HAM10000/output/val/MedSAMAuto_timecost.txt

        # echo "Evaluating on ISIC2016 test..."
        # output_path=$parent_dir/data/ISIC2016/output/test/MedSAMAuto
        # input_path=$parent_dir/data/ISIC2016/input/test/ISIC2016_img
        # seg_path=$parent_dir/data/ISIC2016/input/test/ISIC2016_seg
        # point_path=$parent_dir/data/ISIC2016/input/ISIC2016_test_prompts_42.csv
        # rm -rf $output_path
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id python MedSAMAuto_Inference.py -i $input_path -o $output_path -gt $seg_path
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/ISIC2016/output/test/MedSAMAuto_timecost.txt

        # echo "Evaluating on ISIC2017 test..."
        # output_path=$parent_dir/data/ISIC2017/output/test/MedSAMAuto
        # input_path=$parent_dir/data/ISIC2017/input/test/ISIC2017_img
        # seg_path=$parent_dir/data/ISIC2017/input/test/ISIC2017_seg
        # point_path=$parent_dir/data/ISIC2017/input/ISIC2017_test_prompts_42.csv
        # rm -rf $output_path
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id python MedSAMAuto_Inference.py -i $input_path -o $output_path -gt $seg_path
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/ISIC2017/output/test/MedSAMAuto_timecost.txt

        # echo "Evaluating on PH2 test..."
        # output_path=$parent_dir/data/PH2/output/test/MedSAMAuto
        # input_path=$parent_dir/data/PH2/input/test/PH2_img
        # seg_path=$parent_dir/data/PH2/input/test/PH2_seg
        # point_path=$parent_dir/data/PH2/input/PH2_test_prompts_42.csv
        # rm -rf $output_path
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id python MedSAMAuto_Inference.py -i $input_path -o $output_path -gt $seg_path
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/PH2/output/test/MedSAMAuto_timecost.txt

        # echo "Evaluating on Dermofit test..."
        # output_path=$parent_dir/data/Dermofit/output/test/MedSAMAuto
        # input_path=$parent_dir/data/Dermofit/input/test/Dermofit_img
        # seg_path=$parent_dir/data/Dermofit/input/test/Dermofit_seg
        # point_path=$parent_dir/data/Dermofit/input/Dermofit_test_prompts_42.csv
        # rm -rf $output_path
        # start_time=$(date +%s)
        # CUDA_VISIBLE_DEVICES=$gpu_id python MedSAMAuto_Inference.py -i $input_path -o $output_path -gt $seg_path
        # end_time=$(date +%s)
        # cost_time=$((end_time - start_time))
        # echo "Total cost [$cost_time] seconds" > $parent_dir/data/Dermofit/output/test/MedSAMAuto_timecost.txt
    
        echo "Evaluating on AtlasZoomIn10 test..."
        output_path=$parent_dir/data/AtlasZoomIn10/output/test/MedSAMAuto
        input_path=$parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_img
        seg_path=$parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_seg
        point_path=$parent_dir/data/AtlasZoomIn10/input/AtlasZoomIn10_test_prompts_42.csv
        rm -rf $output_path
        start_time=$(date +%s)
        CUDA_VISIBLE_DEVICES=$gpu_id python MedSAMAuto_Inference.py -i $input_path -o $output_path -gt $seg_path
        end_time=$(date +%s)
        cost_time=$((end_time - start_time))
        echo "Total cost [$cost_time] seconds" > $parent_dir/data/AtlasZoomIn10/output/test/MedSAMAuto_timecost.txt
    fi

    if [ "$calculate_metrics_MedSAMAuto" = true ]; then
        cd $parent_dir/code/compare_models/
        python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/HAM10000/input/val/HAM10000_seg --predict-output $parent_dir/data/HAM10000/output/val/MedSAMAuto --csv-output-path $parent_dir/data/HAM10000/output/val/MedSAMAuto_metrics.csv --image-size $img_resize $img_resize
        python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/HAM10000/input/test/HAM10000_seg --predict-output $parent_dir/data/HAM10000/output/test/MedSAMAuto --csv-output-path $parent_dir/data/HAM10000/output/test/MedSAMAuto_metrics.csv --image-size $img_resize $img_resize
        python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/ISIC2016/input/test/ISIC2016_seg --predict-output $parent_dir/data/ISIC2016/output/test/MedSAMAuto --csv-output-path $parent_dir/data/ISIC2016/output/test/MedSAMAuto_metrics.csv --image-size $img_resize $img_resize
        python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/ISIC2017/input/test/ISIC2017_seg --predict-output $parent_dir/data/ISIC2017/output/test/MedSAMAuto --csv-output-path $parent_dir/data/ISIC2017/output/test/MedSAMAuto_metrics.csv --image-size $img_resize $img_resize
        python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/PH2/input/test/PH2_seg --predict-output $parent_dir/data/PH2/output/test/MedSAMAuto --csv-output-path $parent_dir/data/PH2/output/test/MedSAMAuto_metrics.csv --image-size $img_resize $img_resize
        python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/Dermofit/input/test/Dermofit_seg --predict-output $parent_dir/data/Dermofit/output/test/MedSAMAuto --csv-output-path $parent_dir/data/Dermofit/output/test/MedSAMAuto_metrics.csv --image-size $img_resize $img_resize
        python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_seg --predict-output $parent_dir/data/AtlasZoomIn10/output/test/MedSAMAuto --csv-output-path $parent_dir/data/AtlasZoomIn10/output/test/MedSAMAuto_metrics.csv --image-size $img_resize $img_resize
    fi

    if [ "$upload_MedSAMAuto" = true ]; then
        echo "Checking MedSAMAuto output..."
        cd $parent_dir/code
        python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_val_img --data-path $parent_dir/data/HAM10000/output/val/MedSAMAuto/ --image-size $img_resize $img_resize --image-num 1001
        python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_test_img --data-path $parent_dir/data/HAM10000/output/test/MedSAMAuto/ --image-size $img_resize $img_resize --image-num 1002
        python py_scripts/preprocessing/check_image_size.py --data-name ISIC2016_test_img --data-path $parent_dir/data/ISIC2016/output/test/MedSAMAuto/ --image-size $img_resize $img_resize --image-num 379
        python py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_test_img --data-path $parent_dir/data/ISIC2017/output/test/MedSAMAuto/ --image-size $img_resize $img_resize --image-num 600
        python py_scripts/preprocessing/check_image_size.py --data-name PH2_test_img --data-path $parent_dir/data/PH2/output/test/MedSAMAuto/ --image-size $img_resize $img_resize --image-num 200
        python py_scripts/preprocessing/check_image_size.py --data-name Dermofit_test_img --data-path $parent_dir/data/Dermofit/output/test/MedSAMAuto/ --image-size $img_resize $img_resize --image-num 1300
        # python py_scripts/preprocessing/check_image_size.py --data-name Atlas_test_img --data-path $parent_dir/data/Atlas/output/test/MedSAMAuto/ --image-size $img_resize $img_resize --image-num 960
        python py_scripts/preprocessing/check_image_size.py --data-name AtlasZoomIn10_test_img --data-path $parent_dir/data/AtlasZoomIn10/output/test/MedSAMAuto/ --image-size $img_resize $img_resize --image-num 960
        echo "[√] MedSAMAuto output check pass."

        cd $parent_dir/code/compare_models/
        python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name MedSAMAuto-HAM10000_val --metrics-csv $parent_dir/data/HAM10000/output/val/MedSAMAuto_metrics.csv --timecost-txt $parent_dir/data/HAM10000/output/val/MedSAMAuto_timecost.txt --images-num 1001
        python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name MedSAMAuto-HAM10000_test --metrics-csv $parent_dir/data/HAM10000/output/test/MedSAMAuto_metrics.csv --timecost-txt $parent_dir/data/HAM10000/output/test/MedSAMAuto_timecost.txt --images-num 1002
        python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name MedSAMAuto-ISIC2016_test --metrics-csv $parent_dir/data/ISIC2016/output/test/MedSAMAuto_metrics.csv --timecost-txt $parent_dir/data/ISIC2016/output/test/MedSAMAuto_timecost.txt --images-num 379
        python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name MedSAMAuto-ISIC2017_test --metrics-csv $parent_dir/data/ISIC2017/output/test/MedSAMAuto_metrics.csv --timecost-txt $parent_dir/data/ISIC2017/output/test/MedSAMAuto_timecost.txt --images-num 600
        python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name MedSAMAuto-PH2_test --metrics-csv $parent_dir/data/PH2/output/test/MedSAMAuto_metrics.csv --timecost-txt $parent_dir/data/PH2/output/test/MedSAMAuto_timecost.txt --images-num 200
        python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name MedSAMAuto-Dermofit_test --metrics-csv $parent_dir/data/Dermofit/output/test/MedSAMAuto_metrics.csv --timecost-txt $parent_dir/data/Dermofit/output/test/MedSAMAuto_timecost.txt --images-num 1300
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name MedSAMAuto-Atlas_test --metrics-csv $parent_dir/data/Atlas/output/test/MedSAMAuto_metrics.csv --timecost-txt $parent_dir/data/Atlas/output/test/MedSAMAuto_timecost.txt --images-num 960
        python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name MedSAMAuto-AtlasZoomIn10_test --metrics-csv $parent_dir/data/AtlasZoomIn10/output/test/MedSAMAuto_metrics.csv --timecost-txt $parent_dir/data/AtlasZoomIn10/output/test/MedSAMAuto_timecost.txt --images-num 960
        echo "[√] MedSAMAuto Metrics upload finished."
    fi

    cd $parent_dir/code
    echo "✅ MedSAMAuto compare finished."
    echo "#########################################################################"
fi


############################
#  MedicalSAM              ##
############################
if [ "$COMPARE_MedSAM" = true ]; then
    echo "Comparing with MedicalSAM..."
    mkdir -p $parent_dir/code/compare_models/reps/
    cd $parent_dir/code/compare_models/reps/
    if [ -d "MedicalSAM" ];
    then
        echo "[√] MedicalSAM already exists."
    else
        git clone https://github.com/bowang-lab/MedSAM.git MedicalSAM
        cd $parent_dir/code/compare_models/reps/MedicalSAM
        # pip install -e .
    fi

    # 下载 https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN
    # 到 work_dir/MedSAM/medsam_vit_b

    # cd $parent_dir/code/compare_models/reps/MedicalSAM/work_dir/MedSAM/medsam_vit_b
    # if [ -d "MedicalSAM" ];then
    #     echo "[√] MedicalSAM checkpoint exists."
    # else
    #     wget https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN?usp=drive_link
    #     cd $parent_dir/code/compare_models/reps/MedicalSAM
    # fi
 
    if [ "$retrain_MedSAM" = true ]; then
        # cd $parent_dir/code/py_scripts/preprocessing/

        # # HAM10000
        # python generate_monai_json.py --dataset-name HAM10000 --path $parent_dir/data/HAM10000/input
        # python generate_dataset_csv.py --img-path $parent_dir/data/HAM10000/input/train/HAM10000_img --seg-path $parent_dir/data/HAM10000/input/train/HAM10000_seg --csv-path $parent_dir/data/HAM10000/input/HAM10000_train.csv
        # python generate_dataset_csv.py --img-path $parent_dir/data/HAM10000/input/val/HAM10000_img --seg-path $parent_dir/data/HAM10000/input/val/HAM10000_seg --csv-path $parent_dir/data/HAM10000/input/HAM10000_val.csv
        # python generate_dataset_csv.py --img-path $parent_dir/data/HAM10000/input/test/HAM10000_img --seg-path $parent_dir/data/HAM10000/input/test/HAM10000_seg --csv-path $parent_dir/data/HAM10000/input/HAM10000_test.csv
        
        # # Other test dataset
        # python generate_dataset_csv.py --img-path $parent_dir/data/ISIC2016/input/test/ISIC2016_img --seg-path $parent_dir/data/ISIC2016/input/test/ISIC2016_seg --csv-path $parent_dir/data/ISIC2016/input/ISIC2016_test.csv
        # python generate_dataset_csv.py --img-path $parent_dir/data/ISIC2017/input/test/ISIC2017_img --seg-path $parent_dir/data/ISIC2017/input/test/ISIC2017_seg --csv-path $parent_dir/data/ISIC2017/input/ISIC2017_test.csv
        # python generate_dataset_csv.py --img-path $parent_dir/data/PH2/input/test/PH2_img --seg-path $parent_dir/data/PH2/input/test/PH2_seg --csv-path $parent_dir/data/PH2/input/PH2_test.csv
        # python generate_dataset_csv.py --img-path $parent_dir/data/Atlas/input/test/Atlas_img --seg-path $parent_dir/data/Atlas/input/test/Atlas_seg --csv-path $parent_dir/data/Atlas/input/Atlas_test.csv
        # python generate_dataset_csv.py --img-path $parent_dir/data/Dermofit/input/test/Dermofit_img --seg-path $parent_dir/data/Dermofit/input/test/Dermofit_seg --csv-path $parent_dir/data/Dermofit/input/Dermofit_test.csv

        # cp $parent_dir/code/compare_models/MedSAM-train.py $parent_dir/code/compare_models/reps/MedSAM/train.py
        cd $parent_dir/code/compare_models/reps/MedicalSAM
        CUDA_VISIBLE_DEVICES=$gpu_id python train.py -net sam -mod sam_adpt -exp_name HAM10000 -sam_ckpt $parent_dir/code/pretrained_checkpoint/sam_vit_b_01ec64.pth -image_size 1024 -b 2 -dataset HAM10000 -data_path $parent_dir/data/HAM10000/input -val_freq 1 -vis 1

    fi

    if [ "$evaluate_MedSAM" = true ]; then
        # 需要手动复制pth文件到save路径下
        cd $parent_dir/code/compare_models/reps/MedicalSAM

        echo "Evaluating on HAM10000 test..."
        output_path=$parent_dir/data/HAM10000/output/test/MedSAM
        input_path=$parent_dir/data/HAM10000/input/test/HAM10000_img
        point_path=$parent_dir/data/HAM10000/input/HAM10000_test_prompts_42.csv
        rm -rf $output_path
        start_time=$(date +%s)
        CUDA_VISIBLE_DEVICES=$gpu_id python MedSAM_Inference.py -i $input_path -o $output_path -p $point_path
        end_time=$(date +%s)
        cost_time=$((end_time - start_time))
        echo "Total cost [$cost_time] seconds" > $parent_dir/data/HAM10000/output/test/MedSAM_timecost.txt

        echo "Evaluating on HAM10000 val..."
        output_path=$parent_dir/data/HAM10000/output/val/MedSAM
        input_path=$parent_dir/data/HAM10000/input/val/HAM10000_img
        point_path=$parent_dir/data/HAM10000/input/HAM10000_val_prompts_42.csv
        rm -rf $output_path
        start_time=$(date +%s)
        CUDA_VISIBLE_DEVICES=$gpu_id python MedSAM_Inference.py -i $input_path -o $output_path -p $point_path
        end_time=$(date +%s)
        cost_time=$((end_time - start_time))
        echo "Total cost [$cost_time] seconds" > $parent_dir/data/HAM10000/output/val/MedSAM_timecost.txt

        echo "Evaluating on ISIC2016 test..."
        output_path=$parent_dir/data/ISIC2016/output/test/MedSAM
        input_path=$parent_dir/data/ISIC2016/input/test/ISIC2016_img
        point_path=$parent_dir/data/ISIC2016/input/ISIC2016_test_prompts_42.csv
        rm -rf $output_path
        start_time=$(date +%s)
        CUDA_VISIBLE_DEVICES=$gpu_id python MedSAM_Inference.py -i $input_path -o $output_path -p $point_path
        end_time=$(date +%s)
        cost_time=$((end_time - start_time))
        echo "Total cost [$cost_time] seconds" > $parent_dir/data/ISIC2016/output/test/MedSAM_timecost.txt

        echo "Evaluating on ISIC2017 test..."
        output_path=$parent_dir/data/ISIC2017/output/test/MedSAM
        input_path=$parent_dir/data/ISIC2017/input/test/ISIC2017_img
        point_path=$parent_dir/data/ISIC2017/input/ISIC2017_test_prompts_42.csv
        rm -rf $output_path
        start_time=$(date +%s)
        CUDA_VISIBLE_DEVICES=$gpu_id python MedSAM_Inference.py -i $input_path -o $output_path -p $point_path
        end_time=$(date +%s)
        cost_time=$((end_time - start_time))
        echo "Total cost [$cost_time] seconds" > $parent_dir/data/ISIC2017/output/test/MedSAM_timecost.txt

        echo "Evaluating on PH2 test..."
        output_path=$parent_dir/data/PH2/output/test/MedSAM
        input_path=$parent_dir/data/PH2/input/test/PH2_img
        point_path=$parent_dir/data/PH2/input/PH2_test_prompts_42.csv
        rm -rf $output_path
        start_time=$(date +%s)
        CUDA_VISIBLE_DEVICES=$gpu_id python MedSAM_Inference.py -i $input_path -o $output_path -p $point_path
        end_time=$(date +%s)
        cost_time=$((end_time - start_time))
        echo "Total cost [$cost_time] seconds" > $parent_dir/data/PH2/output/test/MedSAM_timecost.txt

        echo "Evaluating on Dermofit test..."
        output_path=$parent_dir/data/Dermofit/output/test/MedSAM
        input_path=$parent_dir/data/Dermofit/input/test/Dermofit_img
        point_path=$parent_dir/data/Dermofit/input/Dermofit_test_prompts_42.csv
        rm -rf $output_path
        start_time=$(date +%s)
        CUDA_VISIBLE_DEVICES=$gpu_id python MedSAM_Inference.py -i $input_path -o $output_path -p $point_path
        end_time=$(date +%s)
        cost_time=$((end_time - start_time))
        echo "Total cost [$cost_time] seconds" > $parent_dir/data/Dermofit/output/test/MedSAM_timecost.txt
    
        # cd $parent_dir/code/py_scripts/preprocessing/
        # python generate_dataset_csv.py --img-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_img --seg-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_seg --csv-path $parent_dir/data/AtlasZoomIn10/input/AtlasZoomIn10_test.csv

        echo "Evaluating on AtlasZoomIn10 test..."
        output_path=$parent_dir/data/AtlasZoomIn10/output/test/MedSAM
        input_path=$parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_img
        point_path=$parent_dir/data/AtlasZoomIn10/input/AtlasZoomIn10_test_prompts_42.csv
        rm -rf $output_path
        start_time=$(date +%s)
        CUDA_VISIBLE_DEVICES=$gpu_id python MedSAM_Inference.py -i $input_path -o $output_path -p $point_path
        end_time=$(date +%s)
        cost_time=$((end_time - start_time))
        echo "Total cost [$cost_time] seconds" > $parent_dir/data/AtlasZoomIn10/output/test/MedSAM_timecost.txt
    fi

    if [ "$calculate_metrics_MedSAM" = true ]; then
        cd $parent_dir/code/compare_models/
        python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/HAM10000/input/val/HAM10000_seg --predict-output $parent_dir/data/HAM10000/output/val/MedSAM --csv-output-path $parent_dir/data/HAM10000/output/val/MedSAM_metrics.csv --image-size $img_resize $img_resize
        python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/HAM10000/input/test/HAM10000_seg --predict-output $parent_dir/data/HAM10000/output/test/MedSAM --csv-output-path $parent_dir/data/HAM10000/output/test/MedSAM_metrics.csv --image-size $img_resize $img_resize
        python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/ISIC2016/input/test/ISIC2016_seg --predict-output $parent_dir/data/ISIC2016/output/test/MedSAM --csv-output-path $parent_dir/data/ISIC2016/output/test/MedSAM_metrics.csv --image-size $img_resize $img_resize
        python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/ISIC2017/input/test/ISIC2017_seg --predict-output $parent_dir/data/ISIC2017/output/test/MedSAM --csv-output-path $parent_dir/data/ISIC2017/output/test/MedSAM_metrics.csv --image-size $img_resize $img_resize
        python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/PH2/input/test/PH2_seg --predict-output $parent_dir/data/PH2/output/test/MedSAM --csv-output-path $parent_dir/data/PH2/output/test/MedSAM_metrics.csv --image-size $img_resize $img_resize
        python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/Dermofit/input/test/Dermofit_seg --predict-output $parent_dir/data/Dermofit/output/test/MedSAM --csv-output-path $parent_dir/data/Dermofit/output/test/MedSAM_metrics.csv --image-size $img_resize $img_resize
        # python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/Atlas/input/test/Atlas_seg --predict-output $parent_dir/data/Atlas/output/test/MedSAM --csv-output-path $parent_dir/data/Atlas/output/test/MedSAM_metrics.csv --image-size $img_resize $img_resize
        python $parent_dir/code/py_scripts/final/calculate_metrics.py --gt-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_seg --predict-output $parent_dir/data/AtlasZoomIn10/output/test/MedSAM --csv-output-path $parent_dir/data/AtlasZoomIn10/output/test/MedSAM_metrics.csv --image-size $img_resize $img_resize
    fi

    if [ "$upload_MedSAM" = true ]; then
        echo "Checking MedSAM output..."
        cd $parent_dir/code
        python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_val_img --data-path $parent_dir/data/HAM10000/output/val/MedSAM/ --image-size $img_resize $img_resize --image-num 1001
        python py_scripts/preprocessing/check_image_size.py --data-name HAM10000_test_img --data-path $parent_dir/data/HAM10000/output/test/MedSAM/ --image-size $img_resize $img_resize --image-num 1002
        python py_scripts/preprocessing/check_image_size.py --data-name ISIC2016_test_img --data-path $parent_dir/data/ISIC2016/output/test/MedSAM/ --image-size $img_resize $img_resize --image-num 379
        python py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_test_img --data-path $parent_dir/data/ISIC2017/output/test/MedSAM/ --image-size $img_resize $img_resize --image-num 600
        python py_scripts/preprocessing/check_image_size.py --data-name PH2_test_img --data-path $parent_dir/data/PH2/output/test/MedSAM/ --image-size $img_resize $img_resize --image-num 200
        python py_scripts/preprocessing/check_image_size.py --data-name Dermofit_test_img --data-path $parent_dir/data/Dermofit/output/test/MedSAM/ --image-size $img_resize $img_resize --image-num 1300
        # python py_scripts/preprocessing/check_image_size.py --data-name Atlas_test_img --data-path $parent_dir/data/Atlas/output/test/MedSAM/ --image-size $img_resize $img_resize --image-num 960
        python py_scripts/preprocessing/check_image_size.py --data-name AtlasZoomIn10_test_img --data-path $parent_dir/data/AtlasZoomIn10/output/test/MedSAM/ --image-size $img_resize $img_resize --image-num 960
        echo "[√] MedSAM output check pass."

        cd $parent_dir/code/compare_models/
        python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name MedSAM-HAM10000_val --metrics-csv $parent_dir/data/HAM10000/output/val/MedSAM_metrics.csv --timecost-txt $parent_dir/data/HAM10000/output/val/MedSAM_timecost.txt --images-num 1001
        python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name MedSAM-HAM10000_test --metrics-csv $parent_dir/data/HAM10000/output/test/MedSAM_metrics.csv --timecost-txt $parent_dir/data/HAM10000/output/test/MedSAM_timecost.txt --images-num 1002
        python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name MedSAM-ISIC2016_test --metrics-csv $parent_dir/data/ISIC2016/output/test/MedSAM_metrics.csv --timecost-txt $parent_dir/data/ISIC2016/output/test/MedSAM_timecost.txt --images-num 379
        python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name MedSAM-ISIC2017_test --metrics-csv $parent_dir/data/ISIC2017/output/test/MedSAM_metrics.csv --timecost-txt $parent_dir/data/ISIC2017/output/test/MedSAM_timecost.txt --images-num 600
        python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name MedSAM-PH2_test --metrics-csv $parent_dir/data/PH2/output/test/MedSAM_metrics.csv --timecost-txt $parent_dir/data/PH2/output/test/MedSAM_timecost.txt --images-num 200
        python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name MedSAM-Dermofit_test --metrics-csv $parent_dir/data/Dermofit/output/test/MedSAM_metrics.csv --timecost-txt $parent_dir/data/Dermofit/output/test/MedSAM_timecost.txt --images-num 1300
        # python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name MedSAM-Atlas_test --metrics-csv $parent_dir/data/Atlas/output/test/MedSAM_metrics.csv --timecost-txt $parent_dir/data/Atlas/output/test/MedSAM_timecost.txt --images-num 960
        python $parent_dir/code/py_scripts/final/wandb_upload.py --model-name MedSAM-AtlasZoomIn10_test --metrics-csv $parent_dir/data/AtlasZoomIn10/output/test/MedSAM_metrics.csv --timecost-txt $parent_dir/data/AtlasZoomIn10/output/test/MedSAM_timecost.txt --images-num 960
        echo "[√] MedSAM Metrics upload finished."
    fi

    cd $parent_dir/code
    echo "✅ MedSAM compare finished."
    echo "#########################################################################"
fi