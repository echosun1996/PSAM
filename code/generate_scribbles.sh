#!/bin/bash
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

cd $parent_dir/code/py_scripts/scribble
if [ -d "voxynth" ];
then
    echo "[√] voxynth already exists."
else
    git clone https://github.com/dalcalab/voxynth.git voxynth_temp
    mv voxynth_temp/voxynth voxynth
    rm -rf voxynth_temp
fi

# Generate scribbles
# python generate_scribbles.py --dataset-name HAM10000_train --input-path $parent_dir/data/HAM10000/input/train/HAM10000_seg --output-path $parent_dir/data/HAM10000/input/train --labeller $labeller
# python generate_scribbles.py --dataset-name HAM10000_val --input-path $parent_dir/data/HAM10000/input/val/HAM10000_seg --output-path $parent_dir/data/HAM10000/input/val --labeller $labeller
# python generate_scribbles.py --dataset-name HAM10000_test --input-path $parent_dir/data/HAM10000/input/test/HAM10000_seg --output-path $parent_dir/data/HAM10000/input/test --labeller $labeller

# python generate_scribbles.py --dataset-name ISIC2016 --input-path $parent_dir/data/ISIC2016/input/test/ISIC2016_seg --output-path $parent_dir/data/ISIC2016/input/test --labeller $labeller
# python generate_scribbles.py --dataset-name ISIC2017 --input-path $parent_dir/data/ISIC2017/input/test/ISIC2017_seg --output-path $parent_dir/data/ISIC2017/input/test --labeller $labeller
# python generate_scribbles.py --dataset-name PH2 --input-path $parent_dir/data/PH2/input/test/PH2_seg --output-path $parent_dir/data/PH2/input/test --labeller $labeller
# python generate_scribbles.py --dataset-name Dermofit --input-path $parent_dir/data/Dermofit/input/test/Dermofit_seg --output-path $parent_dir/data/Dermofit/input/test --labeller $labeller
# python generate_scribbles.py --dataset-name Atlas --input-path $parent_dir/data/Atlas/input/test/Atlas_seg --output-path $parent_dir/data/Atlas/input/test --labeller $labeller
# python generate_scribbles.py --dataset-name AtlasZoomIn --input-path $parent_dir/data/AtlasZoomIn/input/test/AtlasZoomIn_seg --output-path $parent_dir/data/AtlasZoomIn/input/test --labeller $labeller
# python generate_scribbles.py --dataset-name AtlasZoomIn35 --input-path $parent_dir/data/AtlasZoomIn35/input/test/AtlasZoomIn35_seg --output-path $parent_dir/data/AtlasZoomIn35/input/test --labeller $labeller



# Generate X scribbles
# python generate_x_scribble.py --dataset-name HAM10000_train --input-path $parent_dir/data/HAM10000/input/train/HAM10000_seg --output-path $parent_dir/data/HAM10000/input/train --labeller $labeller
python generate_x_scribble.py --dataset-name HAM10000_val --input-path $parent_dir/data/HAM10000/input/val/HAM10000_seg --output-path $parent_dir/data/HAM10000/input/val --labeller $labeller
python generate_x_scribble.py --dataset-name HAM10000_test --input-path $parent_dir/data/HAM10000/input/test/HAM10000_seg --output-path $parent_dir/data/HAM10000/input/test --labeller $labeller

python generate_x_scribble.py --dataset-name ISIC2016 --input-path $parent_dir/data/ISIC2016/input/test/ISIC2016_seg --output-path $parent_dir/data/ISIC2016/input/test --labeller $labeller
python generate_x_scribble.py --dataset-name ISIC2017 --input-path $parent_dir/data/ISIC2017/input/test/ISIC2017_seg --output-path $parent_dir/data/ISIC2017/input/test --labeller $labeller
python generate_x_scribble.py --dataset-name PH2 --input-path $parent_dir/data/PH2/input/test/PH2_seg --output-path $parent_dir/data/PH2/input/test --labeller $labeller
python generate_x_scribble.py --dataset-name Dermofit --input-path $parent_dir/data/Dermofit/input/test/Dermofit_seg --output-path $parent_dir/data/Dermofit/input/test --labeller $labeller
# python generate_x_scribble.py --dataset-name Atlas --input-path $parent_dir/data/Atlas/input/test/Atlas_seg --output-path $parent_dir/data/Atlas/input/test --labeller $labeller
python generate_x_scribble.py --dataset-name AtlasZoomIn10 --input-path $parent_dir/data/AtlasZoomIn10/input/test/AtlasZoomIn10_seg --output-path $parent_dir/data/AtlasZoomIn10/input/test --labeller $labeller
# python generate_x_scribble.py --dataset-name AtlasZoomIn35 --input-path $parent_dir/data/AtlasZoomIn35/input/test/AtlasZoomIn35_seg --output-path $parent_dir/data/AtlasZoomIn35/input/test --labeller $labeller
