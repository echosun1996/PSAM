import os

# home_directory = os.path.expanduser("~")

import argparse

parser = argparse.ArgumentParser(description="Run nnUNet to predict masks")

parser.add_argument("--preprocess-path", type=str, help="preprocess path")

args = parser.parse_args()

# HAM10000 test
nnUNet_raw = os.path.join(args.preprocess_path)

import multiprocessing
import shutil
from multiprocessing import Pool
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

# from nnunetv2.paths import nnUNet_raw
from skimage import io
from acvl_utils.morphology.morphology_helper import generic_filter_components
from scipy.ndimage import binary_fill_holes
import numpy as np


def load_and_covnert_case(
    input_image: str,
    input_seg: str,
    output_image: str,
    output_seg: str,
    min_component_size: int = 50,
):
    seg = io.imread(input_seg)
    # # unique_values, counts = np.unique(seg, return_counts=True)
    # # distribution = dict(zip(unique_values, counts))
    # # print(distribution)
    # # exit(-1)
    # seg[seg != 0] = 1
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
    dataset_name = "PredictDataset"
    imagestr = join(nnUNet_raw, dataset_name, "imagesTr")
    imagests = join(nnUNet_raw, dataset_name, "imagesTs")
    labelstr = join(nnUNet_raw, dataset_name, "labelsTr")
    labelsts = join(nnUNet_raw, dataset_name, "labelsTs")

    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    train_source = join(nnUNet_raw)

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

        # # test set
        # valid_ids = subfiles(join(test_source, 'output'), join=False, suffix='png')
        # for v in tqdm(valid_ids):
        #     r.append(
        #         p.starmap_async(
        #             load_and_covnert_case,
        #             ((
        #                 join(test_source, 'input', v),
        #                 join(test_source, 'output', v),
        #                 join(imagests, v[:-4] + '_0000.png'),
        #                 join(labelsts, v),
        #                 50
        #             ),)
        #         )
        #     )
        _ = [i.get() for i in r]

    generate_dataset_json(
        join(nnUNet_raw, dataset_name),
        {0: "R", 1: "G", 2: "B"},
        {"background": 0, "lesion": 1},
        num_train,
        ".png",
        dataset_name=dataset_name,
    )


# nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities

# nnUNetv2_predict -i $HOME/zu52_scratch/STI/ISIC_2016_Test/nnUNet_output/PredictDataset/imagesTr -o $HOME/zu52_scratch/STI/ISIC_2016_Test/nnUNet_output/predict -d 666 -c 2d -f 0
# nnUNetv2_predict -i $HOME/zu52_scratch/STI/ISIC_2017_Test_SMALL/nnUNet_output/PredictDataset/imagesTr -o $HOME/zu52_scratch/STI/ISIC_2017_Test_SMALL/nnUNet_output/predict_1 -d 666 -c 2d -f 0
#  nnUNetv2_predict -i $HOME/zu52_scratch/STI/PH2Dataset/nnUNet_output/PredictDataset/imagesTr -o $HOME/zu52_scratch/STI/PH2Dataset/nnUNet_output/predict_1 -d 666 -c 2d -f 0
