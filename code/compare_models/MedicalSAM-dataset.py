""" train and test dataset

author jundewu
"""

import os
import pickle
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from monai.transforms import LoadImage, LoadImaged, Randomizable
from PIL import Image
from skimage import io
from skimage.transform import rotate
from torch.utils.data import Dataset

from utils import random_click


class Dermofit(Dataset):
    def __init__(
        self,
        args,
        data_path,
        transform=None,
        transform_msk=None,
        mode="Training",
        prompt="none",
        plane=False,
    ):

        df = pd.read_csv(
            os.path.join(data_path, "Dermofit_" + mode + ".csv"),
            encoding="gbk",
        )
        self.name_list = df.iloc[:, 1].tolist()
        self.label_list = df.iloc[:, 2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)

        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(msk_path).convert("L")

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == "click":
            point_label, pt = random_click(np.array(mask) / 255, point_label)
        else:
            pt = np.array([0, 0], dtype=np.int32)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)

            if self.transform_msk:
                mask = self.transform_msk(mask)

            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask

        name = name.split("/")[-1].split(".jpg")[0]
        image_meta_dict = {"filename_or_obj": name}
        return {
            "image": img,
            "label": mask,
            "p_label": point_label,
            "pt": pt,
            "image_meta_dict": image_meta_dict,
        }


class AtlasZoomIn10(Dataset):
    def __init__(
        self,
        args,
        data_path,
        transform=None,
        transform_msk=None,
        mode="Training",
        prompt="none",
        plane=False,
    ):

        df = pd.read_csv(
            os.path.join(data_path, "AtlasZoomIn10_" + mode + ".csv"),
            encoding="gbk",
        )
        self.name_list = df.iloc[:, 1].tolist()
        self.label_list = df.iloc[:, 2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)

        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(msk_path).convert("L")

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == "click":
            point_label, pt = random_click(np.array(mask) / 255, point_label)
        else:
            pt = np.array([0, 0], dtype=np.int32)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)

            if self.transform_msk:
                mask = self.transform_msk(mask)

            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask

        name = name.split("/")[-1].split(".jpg")[0]
        image_meta_dict = {"filename_or_obj": name}
        return {
            "image": img,
            "label": mask,
            "p_label": point_label,
            "pt": pt,
            "image_meta_dict": image_meta_dict,
        }


class PH2(Dataset):
    def __init__(
        self,
        args,
        data_path,
        transform=None,
        transform_msk=None,
        mode="Training",
        prompt="none",
        plane=False,
    ):

        df = pd.read_csv(
            os.path.join(data_path, "PH2_" + mode + ".csv"), encoding="gbk"
        )
        self.name_list = df.iloc[:, 1].tolist()
        self.label_list = df.iloc[:, 2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)

        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(msk_path).convert("L")

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == "click":
            point_label, pt = random_click(np.array(mask) / 255, point_label)
        else:
            pt = np.array([0, 0], dtype=np.int32)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)

            if self.transform_msk:
                mask = self.transform_msk(mask)

            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask

        name = name.split("/")[-1].split(".jpg")[0]
        image_meta_dict = {"filename_or_obj": name}
        return {
            "image": img,
            "label": mask,
            "p_label": point_label,
            "pt": pt,
            "image_meta_dict": image_meta_dict,
        }


class ISIC2016(Dataset):
    def __init__(
        self,
        args,
        data_path,
        transform=None,
        transform_msk=None,
        mode="Training",
        prompt="none",
        plane=False,
    ):

        df = pd.read_csv(
            os.path.join(data_path, "ISIC2016_" + mode + ".csv"),
            encoding="gbk",
        )
        self.name_list = df.iloc[:, 1].tolist()
        self.label_list = df.iloc[:, 2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)

        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(msk_path).convert("L")

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == "click":
            point_label, pt = random_click(np.array(mask) / 255, point_label)
        else:
            pt = np.array([0, 0], dtype=np.int32)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)

            if self.transform_msk:
                mask = self.transform_msk(mask)

            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask

        name = name.split("/")[-1].split(".jpg")[0]
        image_meta_dict = {"filename_or_obj": name}
        return {
            "image": img,
            "label": mask,
            "p_label": point_label,
            "pt": pt,
            "image_meta_dict": image_meta_dict,
        }


# class ISIC2016(Dataset):
#     def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):

#         df = pd.read_csv(os.path.join(data_path, 'ISIC2016_' + mode + '_GroundTruth.csv'), encoding='gbk')
#         self.name_list = df.iloc[:,1].tolist()
#         self.label_list = df.iloc[:,2].tolist()
#         self.data_path = data_path
#         self.mode = mode
#         self.prompt = prompt
#         self.img_size = args.image_size

#         self.transform = transform
#         self.transform_msk = transform_msk

#     def __len__(self):
#         return len(self.name_list)

#     def __getitem__(self, index):
#         # if self.mode == 'Training':
#         #     point_label = random.randint(0, 1)
#         #     inout = random.randint(0, 1)
#         # else:
#         #     inout = 1
#         #     point_label = 1
#         point_label = 1

#         """Get the images"""
#         name = self.name_list[index]
#         img_path = os.path.join(self.data_path, name)

#         mask_name = self.label_list[index]
#         msk_path = os.path.join(self.data_path, mask_name)

#         img = Image.open(img_path).convert('RGB')
#         mask = Image.open(msk_path).convert('L')

#         # if self.mode == 'Training':
#         #     label = 0 if self.label_list[index] == 'benign' else 1
#         # else:
#         #     label = int(self.label_list[index])

#         newsize = (self.img_size, self.img_size)
#         mask = mask.resize(newsize)

#         if self.prompt == 'click':
#             point_label, pt = random_click(np.array(mask) / 255, point_label)

#         if self.transform:
#             state = torch.get_rng_state()
#             img = self.transform(img)
#             torch.set_rng_state(state)


#             if self.transform_msk:
#                 mask = self.transform_msk(mask)

#             # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
#             #     mask = 1 - mask

#         name = name.split('/')[-1].split(".jpg")[0]
#         image_meta_dict = {'filename_or_obj':name}
#         return {
#             'image':img,
#             'label': mask,
#             'p_label':point_label,
#             'pt':pt,
#             'image_meta_dict':image_meta_dict,
#         }


class ISIC2017(Dataset):
    def __init__(
        self,
        args,
        data_path,
        transform=None,
        transform_msk=None,
        mode="Training",
        prompt="none",
        plane=False,
    ):

        df = pd.read_csv(
            os.path.join(data_path, "ISIC2017_" + mode + ".csv"),
            encoding="gbk",
        )
        self.name_list = df.iloc[:, 1].tolist()
        self.label_list = df.iloc[:, 2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)

        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(msk_path).convert("L")

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == "click":
            point_label, pt = random_click(np.array(mask) / 255, point_label)
        else:
            pt = np.array([0, 0], dtype=np.int32)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)

            if self.transform_msk:
                mask = self.transform_msk(mask)

            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask

        name = name.split("/")[-1].split(".jpg")[0]
        image_meta_dict = {"filename_or_obj": name}
        return {
            "image": img,
            "label": mask,
            "p_label": point_label,
            "pt": pt,
            "image_meta_dict": image_meta_dict,
        }


class HAM10000(Dataset):
    def __init__(
        self,
        args,
        data_path,
        transform=None,
        transform_msk=None,
        mode="train",
        prompt="none",
        plane=False,
    ):

        df = pd.read_csv(
            os.path.join(data_path, "HAM10000_" + mode + ".csv"),
            encoding="gbk",
        )
        self.name_list = df.iloc[:, 1].tolist()
        self.label_list = df.iloc[:, 2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)

        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(msk_path).convert("L")

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == "click":
            point_label, pt = random_click(np.array(mask) / 255, point_label)
        else:
            pt = np.array([0, 0], dtype=np.int32)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)

            if self.transform_msk:
                mask = self.transform_msk(mask)

            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask

        name = name.split("/")[-1].split(".jpg")[0]
        image_meta_dict = {"filename_or_obj": name}
        return {
            "image": img,
            "label": mask,
            "p_label": point_label,
            "pt": pt,
            "image_meta_dict": image_meta_dict,
        }


# class ISIC2016(Dataset):
#     def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):

#         df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
#         self.name_list = df.iloc[:,1].tolist()
#         self.label_list = df.iloc[:,2].tolist()
#         self.data_path = data_path
#         self.mode = mode
#         self.prompt = prompt
#         self.img_size = args.image_size

#         self.transform = transform
#         self.transform_msk = transform_msk

#     def __len__(self):
#         return len(self.name_list)

#     def __getitem__(self, index):
#         # if self.mode == 'Training':
#         #     point_label = random.randint(0, 1)
#         #     inout = random.randint(0, 1)
#         # else:
#         #     inout = 1
#         #     point_label = 1
#         point_label = 1

#         """Get the images"""
#         name = self.name_list[index]
#         img_path = os.path.join(self.data_path, name)

#         mask_name = self.label_list[index]
#         msk_path = os.path.join(self.data_path, mask_name)

#         img = Image.open(img_path).convert('RGB')
#         mask = Image.open(msk_path).convert('L')

#         # if self.mode == 'Training':
#         #     label = 0 if self.label_list[index] == 'benign' else 1
#         # else:
#         #     label = int(self.label_list[index])

#         newsize = (self.img_size, self.img_size)
#         mask = mask.resize(newsize)

#         if self.prompt == 'click':
#             point_label, pt = random_click(np.array(mask) / 255, point_label)

#         if self.transform:
#             state = torch.get_rng_state()
#             img = self.transform(img)
#             torch.set_rng_state(state)


#             if self.transform_msk:
#                 mask = self.transform_msk(mask)

#             # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
#             #     mask = 1 - mask

#         name = name.split('/')[-1].split(".jpg")[0]
#         image_meta_dict = {'filename_or_obj':name}
#         return {
#             'image':img,
#             'label': mask,
#             'p_label':point_label,
#             'pt':pt,
#             'image_meta_dict':image_meta_dict,
#         }
