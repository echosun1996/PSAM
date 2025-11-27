# Copyright by HQ-SAM team
# All rights reserved.

## data loader
from __future__ import print_function, division
from turtle import pos

import numpy as np
import random
from copy import deepcopy
from skimage import io
import os
from glob import glob
from . import misc
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

#### --------------------- dataloader online ---------------------####


def get_im_gt_name_dict(datasets, args, flag="valid"):
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []

    for i in range(len(datasets)):
        print(
            "--->>>",
            flag,
            " dataset ",
            i,
            "/",
            len(datasets),
            " ",
            datasets[i]["name"],
            "<<<---",
        )
        tmp_im_list, tmp_gt_list = [], []
        tmp_im_list = glob(datasets[i]["im_dir"] + os.sep + "*" + datasets[i]["im_ext"])
        print(datasets[i])
        if args.scribble_type != None:
            tmp_p_scribble_list = glob(datasets[i]["p_scribble"] + os.sep + "*.png")
            tmp_n_scribble_list = glob(datasets[i]["n_scribble"] + os.sep + "*.png")

        print(
            "-im-", datasets[i]["name"], datasets[i]["im_dir"], ": ", len(tmp_im_list)
        )

        if datasets[i]["gt_dir"] == "":
            print(
                "-gt-",
                datasets[i]["name"],
                datasets[i]["gt_dir"],
                ": ",
                "No Ground Truth Found",
            )
            tmp_gt_list = []
        else:
            tmp_gt_list = [
                datasets[i]["gt_dir"]
                + os.sep
                + x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]
                + datasets[i]["gt_ext"]
                for x in tmp_im_list
            ]
            print(
                "-gt-",
                datasets[i]["name"],
                datasets[i]["gt_dir"],
                ": ",
                len(tmp_gt_list),
            )
            if args.scribble_type != None:
                print(
                    "-p_scribble-",
                    datasets[i]["name"],
                    datasets[i]["p_scribble"],
                    ": ",
                    len(tmp_p_scribble_list),
                )
                print(
                    "-n_scribble-",
                    datasets[i]["name"],
                    datasets[i]["n_scribble"],
                    ": ",
                    len(tmp_n_scribble_list),
                )
        if args.scribble_type == None:
            name_im_gt_list.append(
                {
                    "dataset_name": datasets[i]["name"],
                    "im_path": tmp_im_list,
                    "gt_path": tmp_gt_list,
                    "im_ext": datasets[i]["im_ext"],
                    "gt_ext": datasets[i]["gt_ext"],
                }
            )
        else:
            name_im_gt_list.append(
                {
                    "dataset_name": datasets[i]["name"],
                    "im_path": tmp_im_list,
                    "gt_path": tmp_gt_list,
                    "im_ext": datasets[i]["im_ext"],
                    "gt_ext": datasets[i]["gt_ext"],
                    "p_scribble_path": tmp_p_scribble_list,
                    "n_scribble_path": tmp_n_scribble_list,
                }
            )
    return name_im_gt_list


def create_dataloaders(
    name_im_gt_list, my_transforms=[], batch_size=1, training=False, labeller=42
):
    gos_dataloaders = []
    gos_datasets = []

    if len(name_im_gt_list) == 0:
        return gos_dataloaders, gos_datasets

    num_workers_ = 1
    if batch_size > 1:
        num_workers_ = 2
    if batch_size > 4:
        num_workers_ = 4
    if batch_size > 8:
        num_workers_ = 8

    if training:
        for i in range(len(name_im_gt_list)):
            gos_dataset = OnlineDataset(
                [name_im_gt_list[i]],
                transform=transforms.Compose(my_transforms),
                labeller=labeller,
            )
            gos_datasets.append(gos_dataset)

        gos_dataset = ConcatDataset(gos_datasets)
        sampler = DistributedSampler(gos_dataset)
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler, batch_size, drop_last=True
        )
        dataloader = DataLoader(
            gos_dataset, batch_sampler=batch_sampler_train, num_workers=num_workers_
        )

        gos_dataloaders = dataloader
        gos_datasets = gos_dataset

    else:
        for i in range(len(name_im_gt_list)):
            gos_dataset = OnlineDataset(
                [name_im_gt_list[i]],
                transform=transforms.Compose(my_transforms),
                eval_ori_resolution=True,
            )
            sampler = DistributedSampler(gos_dataset, shuffle=False)
            dataloader = DataLoader(
                gos_dataset,
                batch_size,
                sampler=sampler,
                drop_last=False,
                num_workers=num_workers_,
            )

            gos_dataloaders.append(dataloader)
            gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets


class RandomHFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        (
            imidx,
            image,
            label,
            shape,
            points,
            positive_scribble,
            negative_scribble,
        ) = (
            sample["imidx"],
            sample["image"],
            sample["label"],
            sample["shape"],
            sample["points"],
            sample["positive_scribble"],
            sample["negative_scribble"],
        )

        # random horizontal flip
        if random.random() >= self.prob:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[2])
            positive_scribble = torch.flip(positive_scribble, dims=[2])
            negative_scribble = torch.flip(negative_scribble, dims=[2])
            # horizontal flip prompt points
            width = image.shape[1]
            points[:, 0] = width - points[:, 0]

        return {
            "imidx": imidx,
            "image": image,
            "label": label,
            "shape": shape,
            "impath": sample["impath"],
            "points": points,
            "positive_scribble": positive_scribble,
            "negative_scribble": negative_scribble,
        }


class Resize(object):
    def __init__(self, size=[320, 320]):
        self.size = size

    def __call__(self, sample):
        (
            imidx,
            image,
            label,
            shape,
            points,
            positive_scribble,
            negative_scribble,
        ) = (
            sample["imidx"],
            sample["image"],
            sample["label"],
            sample["shape"],
            sample["points"],
            sample["positive_scribble"],
            sample["negative_scribblee"],
        )

        image = torch.squeeze(
            F.interpolate(torch.unsqueeze(image, 0), self.size, mode="bilinear"), dim=0
        )
        label = torch.squeeze(
            F.interpolate(torch.unsqueeze(label, 0), self.size, mode="bilinear"), dim=0
        )
        positive_scribble = torch.squeeze(
            F.interpolate(
                torch.unsqueeze(positive_scribble, 0), self.size, mode="bilinear"
            ),
            dim=0,
        )
        negative_scribble = torch.squeeze(
            F.interpolate(
                torch.unsqueeze(negative_scribble, 0), self.size, mode="bilinear"
            ),
            dim=0,
        )

        return {
            "imidx": imidx,
            "image": image,
            "label": label,
            "shape": torch.tensor(self.size),
            "impath": sample["impath"],
            "positive_scribble": positive_scribble,
            "negative_scribble": negative_scribble,
        }


class RandomCrop(object):
    def __init__(self, size=[288, 288]):
        self.size = size

    def __call__(self, sample):
        imidx, image, label, shape = (
            sample["imidx"],
            sample["image"],
            sample["label"],
            sample["shape"],
        )

        h, w = image.shape[1:]
        new_h, new_w = self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:, top : top + new_h, left : left + new_w]
        label = label[:, top : top + new_h, left : left + new_w]

        return {
            "imidx": imidx,
            "image": image,
            "label": label,
            "shape": torch.tensor(self.size),
            "impath": sample["impath"],
        }


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        imidx, image, label, shape = (
            sample["imidx"],
            sample["image"],
            sample["label"],
            sample["shape"],
        )
        image = normalize(image, self.mean, self.std)

        return {
            "imidx": imidx,
            "image": image,
            "label": label,
            "shape": shape,
            "impath": sample["impath"],
        }


class LargeScaleJitter(object):
    """
    implementation of large scale jitter from copy_paste
    https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py
    """

    def __init__(
        self, output_size=1024, aug_scale_min=0.1, aug_scale_max=2.0, margin_distance=10
    ):
        self.desired_size = torch.tensor(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max
        self.margin_distance = margin_distance

    def pad_target(self, padding, target):
        target = target.copy()
        if "masks" in target:
            target["masks"] = torch.nn.functional.pad(
                target["masks"], (0, padding[1], 0, padding[0])
            )
        return target

    def __call__(self, sample):
        if "positive_scribble" not in sample:
            sample["positive_scribble"] = torch.zeros_like(sample["image"])
        if "negative_scribble" not in sample:
            sample["negative_scribble"] = torch.zeros_like(sample["image"])
        (
            imidx,
            image,
            label,
            image_size,
            points,
            positive_scribble,
            negative_scribble,
        ) = (
            sample["imidx"],
            sample["image"],
            sample["label"],
            sample["shape"],
            sample["points"],
            sample["positive_scribble"],
            sample["negative_scribble"],
        )

        # resize keep ratio
        # out_desired_size = (
        #     (self.desired_size * image_size / max(image_size)).round().int()
        # )

        random_scale = (
            torch.rand(1) * (self.aug_scale_max - self.aug_scale_min)
            + self.aug_scale_min
        )
        scaled_size = (random_scale * self.desired_size).round()

        scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().long()

        scaled_image = torch.squeeze(
            F.interpolate(
                torch.unsqueeze(image, 0), scaled_size.tolist(), mode="bilinear"
            ),
            dim=0,
        )
        scaled_label = torch.squeeze(
            F.interpolate(
                torch.unsqueeze(label, 0), scaled_size.tolist(), mode="bilinear"
            ),
            dim=0,
        )

        scaled_positive_scribble = torch.squeeze(
            F.interpolate(
                torch.unsqueeze(positive_scribble, 0),
                scaled_size.tolist(),
                mode="bilinear",
            ),
            dim=0,
        )

        scaled_negative_scribble = torch.squeeze(
            F.interpolate(
                torch.unsqueeze(negative_scribble, 0),
                scaled_size.tolist(),
                mode="bilinear",
            ),
            dim=0,
        )

        points = (points * scaled_size).round().long()

        # random crop ensuring points are within the crop
        crop_size = (
            min(self.desired_size, scaled_size[0]),
            min(self.desired_size, scaled_size[1]),
        )
        # Ensure crop size does not exceed the scaled image size
        margin_h = max(scaled_size[0] - crop_size[0], 0).item()
        margin_w = max(scaled_size[1] - crop_size[1], 0).item()

        valid_crop = False
        for _ in range(100):  # Try up to 100 times to find a valid crop
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()

            # Check if all points are within the crop
            if (
                (points[:, 0] > crop_y1 + self.margin_distance).all()
                and (points[:, 0] < crop_y2 - self.margin_distance).all()
                and (points[:, 1] > crop_x1 + self.margin_distance).all()
                and (points[:, 1] < crop_x2 - self.margin_distance).all()
            ):
                valid_crop = True
                break

        if not valid_crop:
            # print("Warning: Could not find a valid crop")
            # If no valid crop is found, use the center crop
            offset_h = margin_h // 2
            offset_w = margin_w // 2
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()

        scaled_image = scaled_image[:, crop_y1:crop_y2, crop_x1:crop_x2]
        scaled_label = scaled_label[:, crop_y1:crop_y2, crop_x1:crop_x2]
        scaled_positive_scribble = scaled_positive_scribble[
            :, crop_y1:crop_y2, crop_x1:crop_x2
        ]
        scaled_negative_scribble = scaled_negative_scribble[
            :, crop_y1:crop_y2, crop_x1:crop_x2
        ]

        # crop points
        points[:, 0] -= crop_y1
        points[:, 1] -= crop_x1

        # pad
        padding_h = max(self.desired_size - scaled_image.size(1), 0).item()
        padding_w = max(self.desired_size - scaled_image.size(2), 0).item()
        image = F.pad(scaled_image, [0, padding_w, 0, padding_h], value=128)
        label = F.pad(scaled_label, [0, padding_w, 0, padding_h], value=0)
        positive_scribble = F.pad(
            scaled_positive_scribble, [0, padding_w, 0, padding_h], value=0
        )
        negative_scribble = F.pad(
            scaled_negative_scribble, [0, padding_w, 0, padding_h], value=0
        )

        return {
            "imidx": imidx,
            "image": image,
            "label": label,
            "shape": torch.tensor(image.shape[-2:]),
            "impath": sample["impath"],
            "points": points,
            "positive_scribble": positive_scribble,
            "negative_scribble": negative_scribble,
        }


class OnlineDataset(Dataset):
    def __init__(
        self, name_im_gt_list, transform=None, eval_ori_resolution=False, labeller=42
    ):

        self.transform = transform
        self.dataset = {}
        self.labeller = labeller

        ## combine different datasets into one
        dataset_names = []
        dt_name_list = []  # dataset name per image
        im_name_list = []  # image name
        im_path_list = []  # im path
        gt_path_list = []  # gt path
        positive_scribble_path_list = []  # positive scribble path
        negative_scribble_path_list = []  # negative scribble path

        im_ext_list = []  # im ext
        gt_ext_list = []  # gt ext
        positive_scribble_ext_list = []  # positive scribble ext
        negative_scribble_ext_list = []  # negative scribble ext
        for i in range(0, len(name_im_gt_list)):
            dataset_names.append(name_im_gt_list[i]["dataset_name"])
            # dataset name repeated based on the number of images in this dataset
            dt_name_list.extend(
                [
                    name_im_gt_list[i]["dataset_name"]
                    for x in name_im_gt_list[i]["im_path"]
                ]
            )
            im_name_list.extend(
                [
                    x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0]
                    for x in name_im_gt_list[i]["im_path"]
                ]
            )
            im_path_list.extend(name_im_gt_list[i]["im_path"])
            gt_path_list.extend(name_im_gt_list[i]["gt_path"])
            if "p_scribble_path" in name_im_gt_list[i]:
                positive_scribble_path_list.extend(
                    name_im_gt_list[i]["p_scribble_path"]
                )
            if "n_scribble_path" in name_im_gt_list[i]:
                negative_scribble_path_list.extend(
                    name_im_gt_list[i]["n_scribble_path"]
                )
            im_ext_list.extend(
                [name_im_gt_list[i]["im_ext"] for x in name_im_gt_list[i]["im_path"]]
            )
            gt_ext_list.extend(
                [name_im_gt_list[i]["gt_ext"] for x in name_im_gt_list[i]["gt_path"]]
            )

        self.dataset["data_name"] = dt_name_list
        self.dataset["im_name"] = im_name_list
        self.dataset["im_path"] = im_path_list
        self.dataset["ori_im_path"] = deepcopy(im_path_list)
        self.dataset["gt_path"] = gt_path_list
        self.dataset["positive_scribble_path"] = positive_scribble_path_list
        self.dataset["negative_scribble_path"] = negative_scribble_path_list
        self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
        self.dataset["im_ext"] = im_ext_list
        self.dataset["gt_ext"] = gt_ext_list

        self.eval_ori_resolution = eval_ori_resolution

    def __len__(self):
        return len(self.dataset["im_path"])

    def __getitem__(self, idx):
        im_path = self.dataset["im_path"][idx]
        gt_path = self.dataset["gt_path"][idx]
        if len(self.dataset["positive_scribble_path"]) > 0:
            positive_scribble_path = self.dataset["positive_scribble_path"][idx]
        if len(self.dataset["negative_scribble_path"]) > 0:
            negative_scribble_path = self.dataset["negative_scribble_path"][idx]
        im = io.imread(im_path)
        gt = io.imread(gt_path)
        if len(self.dataset["positive_scribble_path"]) > 0:
            positive_scribble = io.imread(positive_scribble_path)
        if len(self.dataset["negative_scribble_path"]) > 0:
            negative_scribble = io.imread(negative_scribble_path)

        if len(gt.shape) > 2:
            gt = gt[:, :, 0]
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        im = torch.tensor(im.copy(), dtype=torch.float32)
        im = torch.transpose(torch.transpose(im, 1, 2), 0, 1)
        gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32), 0)
        if len(self.dataset["positive_scribble_path"]) > 0:
            positive_scribble = torch.unsqueeze(
                torch.tensor(positive_scribble, dtype=torch.float32), 0
            )
        if len(self.dataset["negative_scribble_path"]) > 0:

            negative_scribble = torch.unsqueeze(
                torch.tensor(negative_scribble, dtype=torch.float32), 0
            )

        points = misc.get_prompt_points(im_path, self.labeller)
        if (
            len(self.dataset["positive_scribble_path"]) > 0
            and len(self.dataset["negative_scribble_path"]) > 0
        ):
            sample = {
                "impath": im_path,
                "imidx": torch.from_numpy(np.array(idx)),
                "image": im,
                "label": gt,
                "shape": torch.tensor(im.shape[-2:]),
                "points": points,
                "positive_scribble": positive_scribble,
                "negative_scribble": negative_scribble,
            }
        elif (
            len(self.dataset["positive_scribble_path"]) > 0
            and len(self.dataset["negative_scribble_path"]) == 0
        ):
            sample = {
                "impath": im_path,
                "imidx": torch.from_numpy(np.array(idx)),
                "image": im,
                "label": gt,
                "shape": torch.tensor(im.shape[-2:]),
                "points": points,
                "positive_scribble": positive_scribble,
                # "negative_scribble": negative_scribble,
            }
        elif (
            len(self.dataset["positive_scribble_path"]) == 0
            and len(self.dataset["negative_scribble_path"]) > 0
        ):
            sample = {
                "impath": im_path,
                "imidx": torch.from_numpy(np.array(idx)),
                "image": im,
                "label": gt,
                "shape": torch.tensor(im.shape[-2:]),
                "points": points,
                # "positive_scribble": positive_scribble,
                "negative_scribble": negative_scribble,
            }
        else:
            sample = {
                "impath": im_path,
                "imidx": torch.from_numpy(np.array(idx)),
                "image": im,
                "label": gt,
                "shape": torch.tensor(im.shape[-2:]),
                "points": points,
            }

        if self.transform:
            sample = self.transform(sample)

        if self.eval_ori_resolution:
            sample["ori_label"] = gt.type(
                torch.uint8
            )  # NOTE for evaluation only. And no flip here
            sample["ori_im_path"] = self.dataset["im_path"][idx]
            sample["ori_gt_path"] = self.dataset["gt_path"][idx]

        return sample
