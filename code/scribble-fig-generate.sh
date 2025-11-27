#!/bin/bash

gpu_sum=3
gpu_id=0,1,2

# Resizing image
split_and_resize=false
img_resize=1024
ham_train_rate=0.8
ham_val_rate=0.1
max_epoch_num=32
# Prompts labeller No.
labeller=42


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




# cd $parent_dir/code/py_scripts/figs
rm -rf $parent_dir/results/scribble_figs/

figs_dir=$parent_dir/results/scribble_figs

# 设置目标目录
target_dir=$parent_dir/results/scribble_figs/temp
if [ ! -d "$target_dir" ]; then
    mkdir -p "$target_dir"
fi


# 文件名初始化
HAM10000_filename="ISIC_0031256.jpg"
ISIC2016_filename="ISIC_0000012.jpg"
ISIC2017_filename="ISIC_0014687.jpg"
Atlas_filename="Atlas_341.jpg"
# Dermofit_filename="D526b.jpg"
Dermofit_filename="B355b.jpg"
PH2_filename="IMD_160.jpg"

# 定义目录和模型数组
dir_directories="HAM10000 ISIC2016 ISIC2017 Dermofit PH2 AtlasZoomIn10" # AtlasZoomIn10
# dir_directories="ISIC2017" # AtlasZoomIn10
# model_directories="PSAM"
# model_directories="PSAM_LinePAndLineN PSAM_XPAndLineN PSAM_XPAndNoN"
# model_directories="PSAM_LinePAndLineN PSAM_LinePAndNoN"
model_directories="PSAM_XPAndNoN"

# 遍历目录数组
for dir in $dir_directories; do
    source_dir="$parent_dir/data/$dir"
    
    # 检查目录是否存在
    if [ -d "$source_dir" ]; then
        case $dir in
            "HAM10000")
                filename_var="$HAM10000_filename"
                ;;
            "AtlasZoomIn10")
                filename_var="$Atlas_filename"
                ;;
            "ISIC2016")
                filename_var="$ISIC2016_filename"
                ;;
            "ISIC2017")
                filename_var="$ISIC2017_filename"
                ;;
            "Dermofit")
                filename_var="$Dermofit_filename"
                ;;
            "PH2")
                filename_var="$PH2_filename"
                ;;
        esac
        # 复制 Ground Truth 和 Original Image
        img_dir="$source_dir/input/test/${dir}_img"
        gt_dir="$source_dir/input/test/${dir}_seg"
        
        if [ -f "$img_dir/$filename_var" ]; then
            cp "$img_dir/$filename_var" "$target_dir/${dir}_Original_${filename_var}"
            cp "$img_dir/$filename_var" "$parent_dir/results/scribble_figs/${dir}_Original_${filename_var}"

        else
            echo "Original image $filename_var does not exist in $img_dir"
        fi

        gt_filename_var="${filename_var%.jpg}_segmentation.png"
        if [ -f "$gt_dir/$gt_filename_var" ]; then
            cp "$gt_dir/$gt_filename_var" "$target_dir/${dir}_GroundTruth_${filename_var}"
        else
            echo "Ground truth $gt_filename_var does not exist in $gt_dir"
        fi

        for model in $model_directories; do
            model_dir="$parent_dir/results/$model/$dir/test/mask"

            
            # 检查文件并复制到目标目录
            if [ -f "$model_dir/$filename_var" ]; then
                new_filename="${dir}_${model}_${filename_var}"
                cp "$model_dir/$filename_var" "$target_dir/$new_filename"

                # add mask to the images
                # echo "Adding mask to the images..."
                source_path=$target_dir/${dir}_Original_${filename_var}
                gt_mask_path=$target_dir/${dir}_GroundTruth_${filename_var}
                pred_mask_path=$target_dir/$new_filename
                save_dir=$figs_dir
                python $parent_dir/code/py_scripts/figs/add_mask.py --source-path $source_path --pred-mask-path $pred_mask_path --gt-mask-path $gt_mask_path --save-dir $save_dir
                
                # # add points to OnePrompt_points, PSAM, SAMSpecific
                # if [ "$model" = "OnePrompt_points" ] || [ "$model" = "PSAM" ] || [ "$model" = "SAMSpecific" ]; then
                #     source_file=$save_dir/$new_filename
                #     points_path=$source_dir/input/${dir}_test_prompts_42.csv
                #     python $parent_dir/code/py_scripts/figs/add_points.py --source-file $source_file --points-path $points_path 
                # fi

                # add LineP 
                if [ "$dir" = "HAM10000" ]; then
                # if $dir == "HAM10000"; then
                    temp="_test"
                else
                    temp=""
                fi

                echo "model: $model"

                if echo "$model" | grep -q "XP"; then
                    source_file=$save_dir/$new_filename
                    scribbles_path=$source_dir/input/test/${dir}${temp}_positive_xScribble_42/${filename_var%.jpg}.png
                    python $parent_dir/code/py_scripts/figs/add_scribbles_to_fig.py --source-file $source_file --scribbles-path $scribbles_path --isPositive True
                fi

                if echo "$model" | grep -q "LineP"; then
                    source_file=$save_dir/$new_filename
                    scribbles_path=$source_dir/input/test/${dir}${temp}_positive_lineScribble_42/${filename_var%.jpg}.png
                    python $parent_dir/code/py_scripts/figs/add_scribbles_to_fig.py --source-file $source_file --scribbles-path $scribbles_path --isPositive True
                fi

                if echo "$model" | grep -q "LineN"; then
                    source_file=$save_dir/$new_filename
                    scribbles_path=$source_dir/input/test/${dir}${temp}_negative_lineScribble_42/${filename_var%.jpg}.png
                    python $parent_dir/code/py_scripts/figs/add_scribbles_to_fig.py --source-file $source_file --scribbles-path $scribbles_path
                fi

            else
                echo "File $filename_var does not exist in $model_dir"
            fi
        done
        
    else
        echo "Directory $source_dir does not exist."
        exit 1
    fi
done
echo "All compare files copied to $target_dir"
rm -rf $target_dir
# Directory where images are saved
figs_dir=$target_dir

# Path to save the final comparison image
# comparison_image_path="comparison_grid.png"

# 调用 Python 程序以拼合图片
# python $parent_dir/code/py_scripts/figs/visual_comparison.py --save-path $comparison_image_path --target-size 256 256


# 生成图例
python $parent_dir/code/py_scripts/scribble/plot_legend.py --output-path $parent_dir/results/scribble_figs/plot_legend.png
