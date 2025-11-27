import argparse
import os
import sys


parser = argparse.ArgumentParser(description="Run nnUNet to predict masks")

parser.add_argument("--data-path", type=str, help="output data folder path")

args = parser.parse_args()


home_directory = os.path.expanduser("~")
source = os.path.join(
    args.data_path,
    "nnUNetFrame/DATASET/nnUNet_raw/nnUNet_cropped_data/Task666_HAM10000",
)
os.environ["nnUNet_raw"] = str(
    os.path.join(
        args.data_path,
        "nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data",
    )
)
os.environ["nnUNet_preprocessed"] = str(
    os.path.join(args.data_path, "nnUNetFrame/DATASET/nnUNet_preprocessed")
)
os.environ["nnUNet_results"] = str(
    os.path.join(args.data_path, "nnUNetFrame/DATASET/nnUNet_trained_models")
)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(
    os.path.join(os.path.dirname(current_dir), "compare_models/reps/nnUNet/")
)


import multiprocessing
import shutil
from multiprocessing import Pool
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import *


from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from skimage import io
import numpy as np


def load_and_covnert_case(
    input_image: str,
    input_seg: str,
    output_image: str,
    output_seg: str,
    min_component_size: int = 50,
):
    seg = io.imread(input_seg)
    # unique_values, counts = np.unique(seg, return_counts=True)
    # distribution = dict(zip(unique_values, counts))
    # print(distribution)
    # exit(-1)
    seg[seg != 0] = 1
    image = io.imread(input_image)
    image = image.sum(2)
    # mask = image == (3 * 255)
    # the dataset has large white areas in which road segmentations can exist but no image information is available.
    # Remove the road label in these areas
    # mask = generic_filter_components(mask, filter_fn=lambda ids, sizes: [i for j, i in enumerate(ids) if
    #                                                                     sizes[j] > min_component_size])
    # mask = binary_fill_holes(mask)
    # seg[mask] = 0
    io.imsave(output_seg, seg, check_contrast=False)
    shutil.copy(input_image, output_image)


if __name__ == "__main__":
    dataset_name = "Dataset666_HAM"
    imagestr = join(nnUNet_raw, dataset_name, "imagesTr")
    imagests = join(nnUNet_raw, dataset_name, "imagesTs")
    labelstr = join(nnUNet_raw, dataset_name, "labelsTr")
    labelsts = join(nnUNet_raw, dataset_name, "labelsTs")

    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    train_source = join(source, "training")
    test_source = join(source, "testing")

    with multiprocessing.get_context("spawn").Pool(8) as p:

        # not all training images have a segmentation
        valid_ids = subfiles(join(train_source, "output"), join=False, suffix="png")
        num_train = len(valid_ids)
        r = []
        for v in tqdm(valid_ids):
            r.append(
                p.starmap_async(
                    load_and_covnert_case,
                    (
                        (
                            join(train_source, "input", v),
                            join(train_source, "output", v),
                            join(imagestr, v[:-4] + "_0000.png"),
                            join(labelstr, v),
                            50,
                        ),
                    ),
                )
            )

        # test set
        valid_ids = subfiles(join(test_source, "output"), join=False, suffix="png")
        for v in tqdm(valid_ids):
            r.append(
                p.starmap_async(
                    load_and_covnert_case,
                    (
                        (
                            join(test_source, "input", v),
                            join(test_source, "output", v),
                            join(imagests, v[:-4] + "_0000.png"),
                            join(labelsts, v),
                            50,
                        ),
                    ),
                )
            )
        _ = [i.get() for i in r]

    generate_dataset_json(
        join(nnUNet_raw, dataset_name),
        {0: "R", 1: "G", 2: "B"},
        {"background": 0, "lesion": 1},
        num_train,
        ".png",
        dataset_name=dataset_name,
    )


# 数据集转换
# nnUNet_convert_decathlon_task -i ~/zu52_scratch/STI/HAM10000/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task666_HAM10000

# export nnUNet_raw_data_base="$HOME/zu52_scratch/STI/HAM10000/nnUNetFrame/DATASET/nnUNet_raw/"
# export nnUNet_preprocessed="$HOME/zu52_scratch/STI/HAM10000/nnUNetFrame/DATASET/nnUNet_preprocessed"
# export RESULTS_FOLDER="$HOME/zu52_scratch/STI/HAM10000/nnUNetFrame/DATASET/nnUNet_trained_models"

# export nnUNet_preprocessed="$HOME/zu52_scratch/seg/nnUNetFrame/DATASET/nnUNet_preprocessed"
# export nnUNet_raw="$HOME/zu52_scratch/seg/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data"
# export nnUNet_results="$HOME/zu52_scratch/seg/nnUNetFrame/DATASET/nnUNet_trained_models"
# 1. run fingerprint extraction, experiment planning and preprocessing
# nnUNetv2_plan_and_preprocess -d 666 --verify_dataset_integrity
# 2. extract the dataset fingerprint of the source dataset
# 10:
# CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 666 2d 0 --c
# 26:
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 666 2d 1 --c
# 27:
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 666 2d 2 --c
# 29:
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 666 2d 3 --c
# 30:X
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 666 2d 4 --c


# Automatically determine the best configuration
