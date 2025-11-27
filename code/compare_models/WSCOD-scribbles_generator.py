import argparse
from operator import index
from tqdm import tqdm

import glob
import math
import random
import sys
import os
import cv2
import numpy as np
import SimpleITK as sitk
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

# from scipy.ndimage import binary_erosion
sys.setrecursionlimit(1000000)
seed = 2022
np.random.seed(seed)
random.seed(seed)


def random_rotation(image, max_angle=15):
    angle = np.random.uniform(-max_angle, max_angle)
    img = Image.fromarray(image)
    img_rotate = img.rotate(angle)
    return img_rotate


def translate_img(img, x_shift, y_shift):

    (height, width) = img.shape[:2]
    matrix = np.float32(np.array([[1, 0, x_shift], [0, 1, y_shift]]))
    trans_img = cv2.warpAffine(img, matrix, (width, height))
    return trans_img


def get_largest_two_component_2D(img, print_info=False, threshold=None):
    """
    Get the largest two components of a binary volume
    inputs:
        img: the input 2D volume
        threshold: a size threshold
    outputs:
        out_img: the output volume
    """
    s = ndimage.generate_binary_structure(2, 2)  # iterate structure
    labeled_array, numpatches = ndimage.label(img, s)  # labeling
    sizes = ndimage.sum(img, labeled_array, range(1, numpatches + 1))
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    if print_info:
        print("component size", sizes_list)
    if len(sizes) == 1:
        out_img = [img]
    else:
        if threshold:
            max_size1 = sizes_list[-1]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            if max_label1.shape[0] > 1:
                max_label1 = max_label1[0]
            component1 = labeled_array == max_label1
            out_img = [component1]
            for temp_size in sizes_list:
                if temp_size > threshold:
                    temp_lab = np.where(sizes == temp_size)[0] + 1
                    temp_cmp = labeled_array == temp_lab[0]
                    out_img.append(temp_cmp)
            return out_img
        else:
            max_size1 = sizes_list[-1]
            max_size2 = sizes_list[-2]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            max_label2 = np.where(sizes == max_size2)[0] + 1
            if max_label1.shape[0] > 1:
                max_label1 = max_label1[0]
            if max_label2.shape[0] > 1:
                max_label2 = max_label2[0]
            component1 = labeled_array == max_label1
            component2 = labeled_array == max_label2
            if max_size2 * 10 > max_size1:
                out_img = [component1, component2]
            else:
                out_img = [component1]
    return out_img


class Cutting_branch(object):
    def __init__(self):
        self.lst_bifur_pt = 0
        self.branch_state = 0
        self.lst_branch_state = 0
        self.direction2delta = {
            0: [-1, -1],
            1: [-1, 0],
            2: [-1, 1],
            3: [0, -1],
            4: [0, 0],
            5: [0, 1],
            6: [1, -1],
            7: [1, 0],
            8: [1, 1],
        }

    def __find_start(self, lab):
        y, x = lab.shape
        idxes = np.asarray(np.nonzero(lab))
        for i in range(idxes.shape[1]):
            pt = tuple([idxes[0, i], idxes[1, i]])
            assert lab[pt] == 1
            directions = []
            for d in range(9):
                if d == 4:
                    continue
                if self.__detect_pt_bifur_state(lab, pt, d):
                    directions.append(d)
            if len(directions) == 1:
                start = pt
                self.start = start
                self.output[start] = 1
                return start
        start = tuple([idxes[0, 0], idxes[1, 0]])
        self.output[start] = 1
        self.start = start
        return start

    def __detect_pt_bifur_state(self, lab, pt, direction):

        d = direction
        y = pt[0] + self.direction2delta[d][0]
        x = pt[1] + self.direction2delta[d][1]
        if lab[y, x] > 0:
            return True
        else:
            return False

    def __detect_neighbor_bifur_state(self, lab, pt):
        directions = []
        for i in range(9):
            if i == 4:
                continue
            if (
                self.output[
                    tuple(
                        [
                            pt[0] + self.direction2delta[i][0],
                            pt[1] + self.direction2delta[i][1],
                        ]
                    )
                ]
                > 0
            ):
                continue
            if self.__detect_pt_bifur_state(lab, pt, i):
                directions.append(i)

        if len(directions) == 0:
            self.end = pt
            return False
        else:
            direction = random.sample(directions, 1)[0]
            next_pt = tuple(
                [
                    pt[0] + self.direction2delta[direction][0],
                    pt[1] + self.direction2delta[direction][1],
                ]
            )
            if len(directions) > 1 and pt != self.start:
                self.lst_output = self.output * 1
                self.previous_bifurPts.append(pt)
            self.output[next_pt] = 1
            pt = next_pt
            self.__detect_neighbor_bifur_state(lab, pt)

    def __detect_loop_branch(self, end):
        for d in range(9):
            if d == 4:
                continue
            y = end[0] + self.direction2delta[d][0]
            x = end[1] + self.direction2delta[d][1]
            if (y, x) in self.previous_bifurPts:
                self.output = self.lst_output * 1
                return True

    def __call__(self, lab, seg_lab, iterations=1):
        self.previous_bifurPts = []
        self.output = np.zeros_like(lab)
        self.lst_output = np.zeros_like(lab)
        components = get_largest_two_component_2D(lab, threshold=15)
        if len(components) > 1:
            for c in components:
                start = self.__find_start(c)
                self.__detect_neighbor_bifur_state(c, start)
        else:
            c = components[0]
            start = self.__find_start(c)
            self.__detect_neighbor_bifur_state(c, start)
        self.__detect_loop_branch(self.end)
        struct = ndimage.generate_binary_structure(2, 2)
        output = ndimage.morphology.binary_dilation(
            self.output, structure=struct, iterations=iterations
        )
        shift_y = random.randint(-6, 6)
        shift_x = random.randint(-6, 6)
        if np.sum(seg_lab) > 1000:
            output = translate_img(output.astype(np.uint8), shift_x, shift_y)
            output = random_rotation(output)
        output = output * seg_lab
        return output


def scrible_2d(label, iteration=[4, 10]):
    if len(label.shape) == 3:
        label = np.mean(label, axis=2)
        label = np.where(label != 0, 1, 0)
    lab = label

    skeleton_map = np.zeros_like(lab, dtype=np.int32)
    struct = ndimage.generate_binary_structure(2, 2)
    if np.sum(lab) > 900 and iteration != 0 and iteration != [0] and iteration != None:
        iter_num = math.ceil(
            iteration[0] + random.random() * (iteration[1] - iteration[0])
        )
        slic = ndimage.morphology.binary_erosion(
            lab, structure=struct, iterations=iter_num
        )
    else:
        slic = lab
    sk_slice = skeletonize(slic, method="lee")
    sk_slice = np.asarray((sk_slice == 255), dtype=np.int32)
    skeleton_map = sk_slice

    return skeleton_map


def scribble4class(label, class_id, class_num, iteration=[4, 10], cut_branch=True):
    label = label == class_id
    sk_map = scrible_2d(label, iteration=iteration)
    if cut_branch and class_id != 0:
        cut = Cutting_branch()
        for i in range(sk_map.shape[0]):
            lab = sk_map[i]
            if lab.sum() < 1:
                continue
            sk_map[i] = cut(lab, seg_lab=label[i])
    if class_id == 0:
        class_id = class_num
    return sk_map * class_id


def generate_scribble(label, iterations, cut_branch=True):
    class_num = np.max(label) + 1
    output = np.zeros_like(label, dtype=np.uint8)
    for i in range(class_num):
        it = iterations[i] if isinstance(iterations, list) else iterations
        scribble = scribble4class(label, i, class_num, it, cut_branch=cut_branch)
        output += scribble.astype(np.uint8)
    return output


if __name__ == "__main__":

    num = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--seg-path", type=str, help="segmentation path")
    parser.add_argument("--visable-out-path", type=str, help="visable output path")
    parser.add_argument("--WSCOD-out-path", type=str, help="WSCOD output")

    args = parser.parse_args()

    # HAM10000
    # img_path = os.path.join(home_path, "zu52_scratch/STI/HAM10000/HAM10000_segmentations/HAM10000_segmentations_lesion_tschandl")
    # out_path = os.path.join(home_path, "zu52_scratch/STI/PSOD_Data/train/Scribble_not/HAM10000") # 原始加点
    # show_out_path = os.path.join(home_path, "zu52_scratch/STI/PSOD_Data/train/scribbles_show") # 显示点
    # WSCOD_out_path = os.path.join(home_path, "zu52_scratch/STI/PSOD_Data/train/Scribble/HAM10000") # 将点加粗作为scibble

    # ISIC 2016
    # img_path = os.path.join(home_path, "zu52_scratch/STI/ISIC_2016_Test/ISBI2016_ISIC_Part1_Test_GroundTruth")
    # out_path = os.path.join(home_path, "zu52_scratch/STI/PSOD_Data/test/ISIC_2016/GT_not")
    # show_out_path = os.path.join(home_path, "zu52_scratch/STI/PSOD_Data/test/ISIC_2016/GT_show")
    # WSCOD_out_path = os.path.join(home_path, "zu52_scratch/STI/PSOD_Data/test/ISIC_2016/GT")

    # ISIC 2017
    seg_path = args.seg_path
    # os.path.join(
    #     home_path,
    #     "zu52_scratch/STI/ISIC_2017_Test_SMALL/ISIC-2017_Test_v2_Part1_GroundTruth",
    # )
    # out_path = os.path.join(
    #     home_path, "zu52_scratch/STI/PSOD_Data/test/ISIC_2017/GT_not"
    # )
    visable_out_path = args.visable_out_path
    # os.path.join(
    #     home_path, "zu52_scratch/STI/PSOD_Data/test/ISIC_2017/GT_vis"
    # )
    WSCOD_out_path = args.WSCOD_out_path
    # os.path.join(
    #     home_path, "zu52_scratch/STI/PSOD_Data/test/ISIC_2017/GT"
    # )

    # PH2
    # img_path = os.path.join(home_path, "zu52_scratch/STI/PH2Dataset/PH2_GroundTruth")
    # out_path = os.path.join(home_path, "zu52_scratch/STI/PSOD_Data/test/PH2/GT_not")
    # show_out_path = os.path.join(home_path, "zu52_scratch/STI/PSOD_Data/test/PH2/GT_show")
    # WSCOD_out_path = os.path.join(home_path, "zu52_scratch/STI/PSOD_Data/test/PH2/GT")

    # Atlas
    # img_path = os.path.join(home_path, "zu52_scratch/STI/AtlasDataset/Atlas_GroundTruth")
    # out_path = os.path.join(home_path, "zu52_scratch/STI/PSOD_Data/test/Atlas/GT_not")
    # show_out_path = os.path.join(home_path, "zu52_scratch/STI/PSOD_Data/test/Atlas/GT_show")
    # WSCOD_out_path = os.path.join(home_path, "zu52_scratch/STI/PSOD_Data/test/Atlas/GT")

    if not os.path.exists(visable_out_path):
        os.makedirs(visable_out_path)
    # if not os.path.exists(out_path):
    #     os.makedirs(out_path)
    if not os.path.exists(WSCOD_out_path):
        os.makedirs(WSCOD_out_path)

    for i in tqdm(
        sorted(glob.glob(str(seg_path) + "/*.png")),
        desc="Processing to: " + str(WSCOD_out_path),
    ):
        # if "ISIC_0000289" not in i:
        #     continue
        itk_data = sitk.ReadImage(i)
        label = sitk.GetArrayFromImage(itk_data)
        # output = scrible_2d(label)

        sk_map_1 = scrible_2d(label)

        label_0 = label == 0
        sk_map_2 = scrible_2d(label_0)
        sk_map_2[sk_map_2 != 0] = 2

        output = np.add(sk_map_1, sk_map_2, dtype=np.int32)

        image = Image.fromarray(output)
        # if str(i).find("_segmentation") != -1:
        #     image.save(
        #         os.path.join(
        #             out_path, i.split("/")[-1].replace("_segmentation.png", ".png")
        #         )
        #     )
        # else:
        #     image.save(
        #         os.path.join(
        #             out_path, i.split("/")[-1].replace("_Segmentation.png", ".png")
        #         )
        #     )

        indices_1 = np.argwhere(output == 1)
        indices_2 = np.argwhere(output == 2)

        # 将每个位置的 x 坐标和 y 坐标分别存储在列表中
        x_1 = [index[1] for index in indices_1]
        y_1 = [index[0] for index in indices_1]

        x_2 = [index[1] for index in indices_2]
        y_2 = [index[0] for index in indices_2]

        # 绘制图像
        plt.imshow(output, cmap="gray")
        plt.axis("off")
        plt.scatter(x_1, y_1, color="red", s=0.1)  # 在值为 1 的位置绘制红色点
        plt.scatter(x_2, y_2, color="blue", s=0.1)  # 在值为 1 的位置绘制红色点
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(
            os.path.join(
                visable_out_path,
                i.split("/")[-1].replace("_segmentation.png", ".png"),
            ),
            bbox_inches="tight",
            pad_inches=0,
        )

        plt.clf()

        im = cv2.imread(i)
        height, width = output.shape[0], output.shape[1]
        output_1 = np.zeros((height, width), dtype=np.uint8)
        for x_i, y_i in zip(x_1, y_1):
            output_1[y_i, x_i] = 255

        output_2 = np.zeros((height, width), dtype=np.uint8)
        for x_i, y_i in zip(x_2, y_2):
            output_2[y_i, x_i] = 255

        # 定义膨胀核
        kernel = np.ones((2, 2), np.uint8)  # 根据需求定义膨胀核的大小
        # 膨胀操作
        dilated_output_1 = cv2.dilate(output_1, kernel, iterations=1)
        dilated_output_2 = cv2.dilate(output_2, kernel, iterations=1)

        dilated_output_1[dilated_output_1 != 0] = 255
        dilated_output_2[dilated_output_2 != 0] = 255

        plt.imshow(dilated_output_1, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(
            os.path.join(
                WSCOD_out_path,
                i.split("/")[-1].replace("_segmentation.png", "_1.png"),
            ),
            bbox_inches="tight",
            pad_inches=0,
        )

        plt.axis("off")
        plt.clf()

        plt.imshow(dilated_output_2, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(
            os.path.join(
                WSCOD_out_path,
                i.split("/")[-1].replace("_segmentation.png", "_2.png"),
            ),
            bbox_inches="tight",  # 这个参数去除图像两侧的黑边
            pad_inches=0,
        )

        plt.axis("off")
        plt.clf()

        covert_out_path = os.path.join(
            WSCOD_out_path, i.split("/")[-1].replace("_Segmentation.png", ".png")
        ).replace("_segmentation.png", ".png")
        covert_out_path_1 = os.path.join(
            WSCOD_out_path, i.split("/")[-1].replace("_Segmentation.png", "_1.png")
        ).replace("_segmentation.png", "_1.png")
        covert_out_path_2 = os.path.join(
            WSCOD_out_path, i.split("/")[-1].replace("_Segmentation.png", "_2.png")
        ).replace("_segmentation.png", "_2.png")

        image_1 = cv2.imread(covert_out_path_1)
        image_1[image_1 != 0] = 1

        image_2 = cv2.imread(covert_out_path_2)
        image_2[image_2 != 0] = 2

        image_final = np.add(image_1, image_2)

        # # 使用numpy的bincount函数计算各个像素值的出现次数
        # histogram = np.bincount(image_final.flatten(), minlength=256)
        # # 输出图像中各个像素值的分布
        # for i, count in enumerate(histogram):
        #     if count > 0:
        #         print(f"Pixel value {i}: {count} times")

        cv2.imwrite(covert_out_path, image_final)
        os.remove(covert_out_path_1)
        os.remove(covert_out_path_2)

        x = None
        y = None
        # print(i)
        # break
        num += 1
    print("end")
