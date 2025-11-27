import os
import argparse
import comm
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple
from PIL import Image
from segment_anything_training import sam_model_registry
from segment_anything_training.modeling import TwoWayTransformer, MaskDecoder

from utils.dataloader import (
    get_im_gt_name_dict,
    create_dataloaders,
    RandomHFlip,
    Resize,
    LargeScaleJitter,
)
from utils.losses import (
    loss_masks,
    loss_masks_whole,
    loss_masks_whole_uncertain,
    loss_boxes,
    loss_uncertain,
    loss_iou,
)
from utils.function import (
    mask_opt,
    save_mask,
    save_timecost,
    show_heatmap,
    show_anns,
    show_heatmap_ax,
    show_anns_ax,
    show_mask,
    show_points,
    show_box,
    show_only_points,
    compute_iou,
    compute_boundary_iou,
)
import utils.misc as misc

from model.mask_decoder_pa import MaskDecoderPA

import logging
import csv
import time
import wandb
import warnings

warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser("PA-SAM", add_help=False)

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the directory where masks and checkpoints will be input",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the directory where masks and checkpoints will be output",
    )
    parser.add_argument(
        "--logfile", type=str, default=None, help="Path to save the log file"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_l",
        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--comments",
        type=str,
        default="",
        help="The comments to add to the wandb run for this experiment",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="The device to run generation on."
    )
    parser.add_argument(
        "--scribble-type",
        type=str,
        default=None,
        help="The type of scribble to use for mask generation.",
    )

    parser.add_argument(
        "--scribble-positive-type",
        type=str,
        default="",
        help="The type to the positive scribble to use for mask generation.",
    )

    parser.add_argument(
        "--prompt_type",
        type=str,
        default="P_B",
        help="The type of prompt to use for mask generation.(P refer to points, B refer to pseudo-box)",
    )

    parser.add_argument(
        "--scribble-negative-type",
        type=str,
        default="",
        help="The type to the negative scribble to use for mask generation.",
    )

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--lr_drop_epoch", default=10, type=int)
    parser.add_argument("--max_epoch_num", default=50, type=int)
    parser.add_argument("--input_size", nargs=2, default=[1024, 1024], type=int)
    parser.add_argument("--batch_size_train", default=5, type=int)
    parser.add_argument("--batch_size_valid", default=1, type=int)
    parser.add_argument("--model_save_fre", default=1, type=int)

    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--labeller", default=-1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", type=int, help="local rank for dist")
    parser.add_argument("--find_unused_params", default=True)

    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--restore-model",
        type=str,
        help="The path to the pa_decoder training checkpoint for evaluation",
    )
    parser.add_argument(
        "--n-sample-points",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--n-negative-points",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--n-positive-points",
        type=int,
        default=4,
    )
    parser.add_argument("--token-visualisation", default=False)
    
    parser.add_argument(
        "--single-image",
        type=str,
        default=None,
        help="Process only a single image (e.g., ISIC_0024985.jpg). If None, process all images."
    )

    return parser.parse_args()


def main(net, train_datasets, valid_datasets, args):

    misc.init_distributed_mode(args)
    print("world size: {}".format(args.world_size))
    print("rank: {}".format(args.rank))
    print("local_rank: {}".format(args.local_rank))
    print("args: " + str(args) + "\n")

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ### --- Step 1: Train or Valid dataset ---
    if not args.eval:
        print("--- create training dataloader ---")
        train_im_gt_list = get_im_gt_name_dict(train_datasets, args, flag="train")
        train_dataloaders, train_datasets = create_dataloaders(
            train_im_gt_list,
            my_transforms=[LargeScaleJitter()],  # RandomHFlip(),
            batch_size=args.batch_size_train,
            training=True,
            labeller=args.labeller,
        )
        print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, args, flag="valid")
    
    # Filter to single image if specified
    if args.single_image is not None:
        print(f"Filtering to single image: {args.single_image}")
        single_image_basename = os.path.splitext(args.single_image)[0]
        
        for dataset_dict in valid_im_gt_list:
            # Filter image paths
            filtered_im_paths = [
                im_path for im_path in dataset_dict["im_path"]
                if os.path.basename(im_path) == args.single_image or 
                   os.path.splitext(os.path.basename(im_path))[0] == single_image_basename
            ]
            
            if len(filtered_im_paths) > 0:
                # Update image paths
                dataset_dict["im_path"] = filtered_im_paths
                
                # Update corresponding ground truth paths
                if len(dataset_dict["gt_path"]) > 0:
                    dataset_dict["gt_path"] = [
                        os.path.dirname(dataset_dict["gt_path"][0])
                        + os.sep
                        + os.path.splitext(os.path.basename(im_path))[0]
                        + dataset_dict["gt_ext"]
                        for im_path in filtered_im_paths
                    ]
                
                # Update scribble paths if they exist
                if "p_scribble_path" in dataset_dict and len(dataset_dict["p_scribble_path"]) > 0:
                    dataset_dict["p_scribble_path"] = [
                        sp for sp in dataset_dict["p_scribble_path"]
                        if single_image_basename in os.path.basename(sp)
                    ]
                
                if "n_scribble_path" in dataset_dict and len(dataset_dict["n_scribble_path"]) > 0:
                    dataset_dict["n_scribble_path"] = [
                        sp for sp in dataset_dict["n_scribble_path"]
                        if single_image_basename in os.path.basename(sp)
                    ]
                
                print(f"Found {len(filtered_im_paths)} matching image(s) in {dataset_dict['dataset_name']}")
            else:
                # Remove datasets that don't contain the specified image
                dataset_dict["im_path"] = []
                dataset_dict["gt_path"] = []
                if "p_scribble_path" in dataset_dict:
                    dataset_dict["p_scribble_path"] = []
                if "n_scribble_path" in dataset_dict:
                    dataset_dict["n_scribble_path"] = []
        
        # Remove empty datasets
        valid_im_gt_list = [d for d in valid_im_gt_list if len(d["im_path"]) > 0]
        
        if len(valid_im_gt_list) == 0:
            print(f"Warning: Image {args.single_image} not found in any validation dataset!")
    
    valid_dataloaders, valid_datasets = create_dataloaders(
        valid_im_gt_list,
        my_transforms=[],  # [Resize(args.input_size)],
        batch_size=args.batch_size_valid,
        training=False,
    )
    print(len(valid_dataloaders), " valid dataloaders created")

    ### --- Step 2: DistributedDataParallel---
    if torch.cuda.is_available():
        net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params
    )
    net_without_ddp = net.module

    ### --- Step 3: Train or Evaluate ---
    if not args.eval:
        print("--- define optimizer ---")
        optimizer = optim.Adam(
            net_without_ddp.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
        lr_scheduler.last_epoch = args.start_epoch

        train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler)
    else:
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
        _ = sam.to(device=args.device)
        sam = torch.nn.parallel.DistributedDataParallel(
            sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params
        )

        if args.restore_model:
            print("restore model from:", args.restore_model)
            if torch.cuda.is_available():
                net_without_ddp.load_state_dict(
                    torch.load(args.restore_model), strict=False
                )
            else:
                net_without_ddp.load_state_dict(
                    torch.load(args.restore_model, map_location="cpu")
                )

        evaluate(args, net, sam, valid_dataloaders, args.visualize, print_func=print)


def train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler):
    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)
        if not args.logfile:
            args.logfile = args.output + "/" + args.output[10:] + "_train.txt"
        if os.path.exists(args.logfile):
            os.remove(args.logfile)
        logging.basicConfig(filename=args.logfile, level=logging.INFO)

    def print(*args, **kwargs):
        output = " ".join(str(arg) for arg in args)
        logging.info(output)
        built_in_print(*args, **kwargs)

    built_in_print = __builtins__.print

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num
    train_num = len(train_dataloaders)

    net.train()
    _ = net.to(device=args.device)

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    sam = torch.nn.parallel.DistributedDataParallel(
        sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params
    )

    for epoch in range(epoch_start, epoch_num):
        print("epoch:   ", epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
        os.environ["CURRENT_EPOCH"] = str(epoch)
        metric_logger = misc.MetricLogger(delimiter="  ")
        train_dataloaders.batch_sampler.sampler.set_epoch(epoch)

        for data in metric_logger.log_every(
            train_dataloaders, 20, logger=args.logfile, print_func=print
        ):

            inputs, labels = data["image"], data["label"]
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            device = (
                inputs.device
                if isinstance(inputs, torch.Tensor)
                else torch.device("cpu")
            )
            imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()

            # input prompt
            if args.scribble_type == None:
                input_keys = [
                    "box",
                    "point",
                    "noise_mask",
                    "box+point",
                    "box+noise_mask",
                    "point+noise_mask",
                    "box+point+noise_mask",
                ]
                labels_points = misc.masks_sample_points(labels[:, 0, :, :])
                labels_box = misc.points_to_fake_boxes(labels_points)
            else:
                input_keys = [
                    "box",
                    "scribble",
                    "noise_mask",
                    "box+scribble",
                    "box+noise_mask",
                    "scribble+noise_mask",
                    "box+scribble+noise_mask",
                ]
                positive = []
                negative = []
                if (
                    args.scribble_positive_type != None
                    and args.scribble_positive_type != ""
                ):
                    positive_scribbles = data["positive_scribble"]
                    positive = misc.get_points_from_scribbles(
                        positive_scribbles[:, 0, :, :]
                    )
                if (
                    args.scribble_negative_type != None
                    and args.scribble_negative_type != ""
                ):
                    negative_scribbles = data["negative_scribble"]
                    negative = misc.get_points_from_scribbles(
                        negative_scribbles[:, 0, :, :]
                    )
                labels_box = misc.points_to_fake_boxes(positive_scribbles).to(device)

            impath = data["impath"]

            labels_256 = F.interpolate(labels, size=(256, 256), mode="bilinear")
            labels_noisemask = misc.masks_noise(labels_256)

            batched_input = []
            gt_boxes = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = (
                    torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device)
                    .permute(2, 0, 1)
                    .contiguous()
                )
                dict_input["image"] = input_image
                input_type = random.choice(input_keys)
                gt_boxes.append(
                    (labels_box[b_i : b_i + 1] / 1024).clamp(min=0.0, max=1.0)
                )
                # noise_box = misc.box_noise(labels_box[b_i : b_i + 1], box_noise_scale=1)
                if "scribble" in input_type:
                    dict_input["point_labels"] = torch.empty(0, device=device)
                    if positive and len(positive) > 0:
                        point_coords_p = positive[b_i : b_i + 1]
                        dict_input["point_labels"] = torch.ones(
                            len(point_coords_p), device=device
                        )[None, :]
                    if negative and len(negative) > 0:
                        point_coords_n = negative[b_i : b_i + 1]
                        if len(point_coords_n) > 0:
                            # 如果 negative 存在，添加负标签
                            if (
                                dict_input["point_labels"].size(1) > 0
                            ):  # 如果已经有正标签，拼接负标签
                                dict_input["point_labels"] = torch.cat(
                                    (
                                        dict_input["point_labels"],
                                        torch.zeros(len(point_coords_n), device=device)[
                                            None, :
                                        ],
                                    ),
                                    dim=1,
                                )
                            else:
                                # 如果没有正标签，直接创建负标签
                                dict_input["point_labels"] = torch.zeros(
                                    len(point_coords_n), device=device
                                )[None, :]
                        # point_coords_n = negative[b_i : b_i + 1]
                        # dict_input["point_labels"] = torch.cat(
                        #     (
                        #         dict_input["point_labels"],
                        #         torch.zeros(len(point_coords_n), device=device)[None, :],
                        #     ),
                        #     dim=1,
                        # ).to(device)
                    dict_input["point_labels"] = dict_input["point_labels"].to(device)

                if "box" in input_type:
                    dict_input["boxes"] = labels_box[b_i : b_i + 1].to(device)
                if "point" in input_type:
                    point_coords = labels_points[b_i : b_i + 1]
                    dict_input["point_coords"] = point_coords
                    dict_input["point_labels"] = torch.ones(
                        point_coords.shape[1], device=sam.device
                    )[None, :]

                if "noise_mask" in input_type:
                    dict_input["mask_inputs"] = labels_noisemask[b_i : b_i + 1]
                # else:
                #     raise NotImplementedError
                dict_input["original_size"] = imgs[b_i].shape[:2]
                dict_input["label"] = labels[b_i : b_i + 1]
                batched_input.append(dict_input)

                # print("Device of sam:", sam.device)
                # print("Device of dict_input:", dict_input["point_labels"].device)
            with torch.no_grad():
                batched_output, interm_embeddings = (
                    sam.module.forward_for_prompt_adapter(
                        batched_input, multimask_output=False
                    )
                )

            gt_boxes = torch.cat(gt_boxes, 0)
            batch_len = len(batched_output)
            encoder_embedding = torch.cat(
                [batched_output[i_l]["encoder_embedding"] for i_l in range(batch_len)],
                dim=0,
            )
            image_pe = [batched_output[i_l]["image_pe"] for i_l in range(batch_len)]
            sparse_embeddings = [
                batched_output[i_l]["sparse_embeddings"] for i_l in range(batch_len)
            ]
            dense_embeddings = [
                batched_output[i_l]["dense_embeddings"] for i_l in range(batch_len)
            ]
            image_record = [
                batched_output[i_l]["image_record"] for i_l in range(batch_len)
            ]
            input_images = batched_output[0]["input_images"]
            (
                masks_sam,
                iou_preds,
                uncertain_maps,
                final_masks,
                coarse_masks,
                refined_masks,
                box_preds,
                ref_maps,
                ps,
                ns,
            ) = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                interm_embeddings=interm_embeddings,
                image_record=image_record,
                prompt_encoder=sam.module.prompt_encoder,
                input_images=input_images,
                n_sample_points=args.n_sample_points,
                n_positive_points=args.n_positive_points,
                n_negative_points=args.n_negative_points,
            )

            loss_mask, loss_dice = loss_masks_whole(
                masks_sam, labels / 255.0, len(masks_sam)
            )
            loss = loss_mask + loss_dice

            loss_mask_final, loss_dice_final = loss_masks_whole_uncertain(
                coarse_masks,
                refined_masks,
                labels / 255.0,
                uncertain_maps,
                len(final_masks),
            )
            loss = loss + (loss_mask_final + loss_dice_final)
            loss_uncertain_map, gt_uncertain = loss_uncertain(uncertain_maps, labels)
            loss = loss + loss_uncertain_map

            loss_dict = {
                "loss_mask": loss_mask,
                "loss_dice": loss_dice,
                "loss_mask_final": loss_mask_final,
                "loss_dice_final": loss_dice_final,
                "loss_uncertain_map": loss_uncertain_map,
            }

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            losses_reduced_scaled = sum(loss_dict_reduced.values())
            loss_value = losses_reduced_scaled.item()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)

        print("Finished epoch:      ", epoch)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {
            k: meter.global_avg
            for k, meter in metric_logger.meters.items()
            if meter.count > 0
        }
        print("train_stats:", train_stats)

        lr_scheduler.step()
        test_stats = evaluate(args, net, sam, valid_dataloaders, print_func=print)
        print("test_stats:", test_stats)

        train_stats.update(test_stats)
        if not args.eval:
            print("Upload to wandb...")
            wandb.log(train_stats)
        net.train()

        if epoch % args.model_save_fre == 0:
            model_name = "/epoch_" + str(epoch) + ".pth"
            print("come here save at", args.output + model_name)
            misc.save_on_master(net.module.state_dict(), args.output + model_name)

    # Finish training
    print("Training Reaches The Maximum Epoch Number")

    # merge sam and pa_decoder
    if misc.is_main_process():
        sam_ckpt = torch.load(args.checkpoint)
        pa_decoder = torch.load(args.output + model_name)
        sam_ckpt.update(
            {
                k.replace("mask_decoder", "mask_decoder_ori"): v
                for k, v in sam_ckpt.items()
                if "mask_decoder" in k
            }
        )
        for key in pa_decoder.keys():
            sam_key = "mask_decoder." + key
            sam_ckpt[sam_key] = pa_decoder[key]
        model_name = "/sam_pa_epoch_" + str(epoch) + ".pth"
        torch.save(sam_ckpt, args.output + model_name)


def evaluate(args, net, sam, valid_dataloaders, visualize=False, print_func=print):

    print = print_func

    if args.eval and not args.visualize:
        if not args.logfile:
            args.logfile = args.output + "/" + args.output[10:] + "_eval.txt"
        if os.path.exists(args.logfile):
            os.remove(args.logfile)
        logging.basicConfig(filename=args.logfile, level=logging.INFO)

        def print(*args, **kwargs):
            output = " ".join(str(arg) for arg in args)
            logging.info(output)
            built_in_print(*args, **kwargs)

        built_in_print = __builtins__.print

    net.eval()
    print("Validating...")
    test_stats = {}

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print("valid_dataloader len:", len(valid_dataloader))

        iou_result = []
        biou_result = []
        img_id = []
        dataset_name = ["DIS", "COIFT", "HRSOD", "ThinObject"]
        total_time = 0

        for data_val in metric_logger.log_every(
            valid_dataloader, 1000, logger=args.logfile, print_func=print
        ):
            start_time = time.time()
            imidx_val, inputs_val, labels_val, shapes_val, labels_ori = (
                data_val["imidx"],
                data_val["image"],
                data_val["label"],
                data_val["shape"],
                data_val["ori_label"],
            )

            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_val = labels_val.cuda()
                labels_ori = labels_ori.cuda()
            device = (
                inputs_val.device
                if isinstance(inputs_val, torch.Tensor)
                else torch.device("cpu")
            )
            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()

            # labels_box = misc.masks_to_boxes(labels_val[:, 0, :, :])
            # input_keys = ["box"]

            if args.scribble_type == None:
                input_keys = ["box+point"]
                labels_points = data_val["points"].to(device)
                # on_lesion = misc.check_points_on_lesion(labels_points, labels_val)
                # if not on_lesion:
                #     print("^^^^^^^eva^^^^^^^^^")
                #     print("eva labels_points:", labels_points)
                #     misc.show_points_on_inputs(labels_points, inputs_val)
                #     exit(-1)
                labels_box = misc.points_to_fake_boxes(labels_points)

            else:
                input_keys = ["box+scribble"]

                positive = []
                negative = []
                if (
                    args.scribble_positive_type != None
                    and args.scribble_positive_type != ""
                ):
                    positive_scribbles = data_val["positive_scribble"]
                    positive = misc.get_points_from_scribbles(
                        positive_scribbles[:, 0, :, :]
                    )
                if (
                    args.scribble_negative_type != None
                    and args.scribble_negative_type != ""
                ):
                    negative_scribbles = data_val["negative_scribble"]
                    negative = misc.get_points_from_scribbles(
                        negative_scribbles[:, 0, :, :]
                    )

                # positive_scribbles = data_val["positive_scribble"]
                # negative_scribbles = data_val["negative_scribble"]
                # positive = misc.get_points_from_scribbles(
                #     positive_scribbles[:, 0, :, :]
                # )
                # negative = misc.get_points_from_scribbles(
                #     negative_scribbles[:, 0, :, :]
                # )
                labels_box = misc.points_to_fake_boxes(positive_scribbles).to(device)

            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = (
                    torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device)
                    .permute(2, 0, 1)
                    .contiguous()
                )
                dict_input["image"] = input_image
                input_type = random.choice(input_keys)
                if args.prompt_type == "P":
                    input_type = "point"
                elif args.prompt_type == "B":
                    input_type = "box"

                if "scribble" in input_type:
                    dict_input["point_labels"] = torch.empty(0, device=device)
                    if positive and len(positive) > 0:
                        point_coords_p = positive[b_i : b_i + 1]
                        dict_input["point_labels"] = torch.ones(
                            len(point_coords_p), device=device
                        )[None, :]
                    if negative and len(negative) > 0:
                        point_coords_n = negative[b_i : b_i + 1]
                        if len(point_coords_n) > 0:
                            # 如果 negative 存在，添加负标签
                            if (
                                dict_input["point_labels"].size(1) > 0
                            ):  # 如果已经有正标签，拼接负标签
                                dict_input["point_labels"] = torch.cat(
                                    (
                                        dict_input["point_labels"],
                                        torch.zeros(len(point_coords_n), device=device)[
                                            None, :
                                        ],
                                    ),
                                    dim=1,
                                )
                            else:
                                # 如果没有正标签，直接创建负标签
                                dict_input["point_labels"] = torch.zeros(
                                    len(point_coords_n), device=device
                                )[None, :]

                    # point_coords_p = positive[b_i : b_i + 1]
                    # # dict_input["point_coords"] = point_coords_p
                    # dict_input["point_labels"] = torch.ones(
                    #     len(point_coords_p), device=device
                    # )[None, :]

                    # point_coords_n = negative[b_i : b_i + 1]
                    # dict_input["point_labels"] = torch.cat(
                    #     (
                    #         dict_input["point_labels"],
                    #         torch.zeros(len(point_coords_n), device=device)[None, :],
                    #     ),
                    #     dim=1,
                    # ).to(device)
                if "box" in input_type:
                    dict_input["boxes"] = labels_box[b_i : b_i + 1]
                if args.prompt_type == "B":
                    # add negative points.
                    corner_coords = torch.tensor(
                        [
                            [
                                [0, 0],
                                [0, args.input_size[1]],
                                [args.input_size[0], 0],
                                [args.input_size[0], args.input_size[1]],
                            ]
                        ],
                        device=device,
                    )
                    corner_labels = torch.zeros(
                        corner_coords.shape[1], device=corner_coords.device
                    )[None, :]
                    dict_input["point_coords"] = corner_coords.to(device)
                    dict_input["point_labels"] = corner_labels.to(device)

                if "point" in input_type:
                    point_coords = labels_points[b_i : b_i + 1]
                    dict_input["point_coords"] = point_coords
                    dict_input["point_labels"] = torch.ones(
                        point_coords.shape[1], device=device
                    )[None, :]
                    # add negative points.
                    corner_coords = torch.tensor(
                        [
                            [
                                [0, 0],
                                [0, args.input_size[1]],
                                [args.input_size[0], 0],
                                [args.input_size[0], args.input_size[1]],
                            ]
                        ],
                        device=device,
                    )
                    corner_labels = torch.zeros(
                        corner_coords.shape[1], device=corner_coords.device
                    )[None, :]
                    dict_input["point_coords"] = torch.cat(
                        (dict_input["point_coords"], corner_coords), dim=1
                    ).to(device)
                    dict_input["point_labels"] = torch.cat(
                        (dict_input["point_labels"], corner_labels), dim=1
                    ).to(device)

                # if "noise_mask" in input_type:
                #     dict_input["mask_inputs"] = labels_noisemask[b_i : b_i + 1]
                # else:
                #     raise NotImplementedError
                dict_input["original_size"] = imgs[b_i].shape[:2]
                dict_input["label"] = data_val["label"][b_i : b_i + 1]
                batched_input.append(dict_input)

            with torch.no_grad():
                batched_output, interm_embeddings = (
                    sam.module.forward_for_prompt_adapter(
                        batched_input, multimask_output=False
                    )
                )

            batch_len = len(batched_output)
            encoder_embedding = torch.cat(
                [batched_output[i_l]["encoder_embedding"] for i_l in range(batch_len)],
                dim=0,
            )
            image_pe = [batched_output[i_l]["image_pe"] for i_l in range(batch_len)]
            sparse_embeddings = [
                batched_output[i_l]["sparse_embeddings"] for i_l in range(batch_len)
            ]
            dense_embeddings = [
                batched_output[i_l]["dense_embeddings"] for i_l in range(batch_len)
            ]
            image_record = [
                batched_output[i_l]["image_record"] for i_l in range(batch_len)
            ]
            input_images = batched_output[0]["input_images"]
            (
                masks_sam,
                iou_preds,
                uncertain_maps,
                final_masks,
                coarse_masks,
                refined_masks,
                box_preds,
                ref_maps,
                ps,
                ns,
            ) = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                interm_embeddings=interm_embeddings,
                image_record=image_record,
                prompt_encoder=sam.module.prompt_encoder,
                input_images=input_images,
                n_sample_points=args.n_sample_points,
                n_positive_points=args.n_positive_points,
                n_negative_points=args.n_negative_points,
            )

            iou = compute_iou(masks_sam, labels_ori)
            boundary_iou = compute_boundary_iou(masks_sam, labels_ori)
            end_time = time.time()
            timecost = end_time - start_time
            if visualize:
                print("visualize")
                os.makedirs(args.output, exist_ok=True)
                masks_pa_vis = (
                    F.interpolate(
                        masks_sam.detach(),
                        (1024, 1024),
                        mode="bilinear",
                        align_corners=False,
                    )
                    > 0
                ).cpu()

                if args.token_visualisation:
                    dense_vis = (
                        F.interpolate(
                            (uncertain_maps >= 0.5).detach() * refined_masks,
                            (1024, 1024),
                            mode="bilinear",
                            align_corners=False,
                        )
                    ).cpu()
                    scaled_masks = (
                        (dense_vis - torch.min(dense_vis))
                        / (torch.max(dense_vis) - torch.min(dense_vis))
                        * 255
                    )
                    dense_vis = scaled_masks.type(torch.uint8)

                    spare_vis = (
                        F.interpolate(
                            (uncertain_maps < 0.5).detach() * coarse_masks,
                            (1024, 1024),
                            mode="bilinear",
                            align_corners=False,
                        )
                    ).cpu()
                    scaled_masks = (
                        (spare_vis - torch.min(spare_vis))
                        / (torch.max(spare_vis) - torch.min(spare_vis))
                        * 255
                    )
                    spare_vis = scaled_masks.type(torch.uint8)

                    ref_map_vis = (
                        F.interpolate(
                            (ref_maps < 0.5).detach() * coarse_masks,
                            (1024, 1024),
                            mode="bilinear",
                            align_corners=False,
                        )
                    ).cpu()
                    scaled_masks = (
                        (ref_map_vis - torch.min(ref_map_vis))
                        / (torch.max(ref_map_vis) - torch.min(ref_map_vis))
                        * 255
                    )
                    ref_map_vis = scaled_masks.type(torch.uint8)

                    masks_pa_vis = (
                        F.interpolate(
                            masks_sam.detach(),
                            (1024, 1024),
                            mode="bilinear",
                            align_corners=False,
                        )
                        > 0
                    ).cpu()
                    coarse_masks_vis = (
                        F.interpolate(
                            coarse_masks.detach(),
                            (1024, 1024),
                            mode="bilinear",
                            align_corners=False,
                        )
                        > 0
                    ).cpu()
                    refined_masks_vis = (
                        F.interpolate(
                            refined_masks.detach(),
                            (1024, 1024),
                            mode="bilinear",
                            align_corners=False,
                        )
                        > 0
                    ).cpu()
                    uncertain_maps_vis = (
                        F.interpolate(
                            uncertain_maps.detach(),
                            (1024, 1024),
                            mode="bilinear",
                            align_corners=False,
                        )
                    ).cpu()
                    scaled_masks = (
                        (uncertain_maps_vis - torch.min(uncertain_maps_vis))
                        / (
                            torch.max(uncertain_maps_vis)
                            - torch.min(uncertain_maps_vis)
                        )
                        * 255
                    )
                    uncertain_maps_vis = scaled_masks.type(torch.uint8)

                    # 正反例点
                    ps_vis = (ps.detach() > 0).cpu()
                    ns_vis = (ns.detach() > 0).cpu()

                for ii in range(len(imgs)):
                    # data_val["ori_im_path"]
                    # ~/zu52/PSAM/data/HAM10000/input/val/HAM10000_img/ISIC_0031526.jpg
                    img_path = data_val["ori_im_path"][ii]
                    img_name = img_path.split("/")[-1]
                    data_name = img_path.split("/")[-5]  # HAM10000
                    date_class = img_path.split("/")[-3]  # val

                    save_path = os.path.join(args.output, data_name, date_class)

                    timecost_path = os.path.join(save_path, "timecost.csv")
                    vis_path = os.path.join(save_path, "vis")
                    mask_path = os.path.join(save_path, "mask")

                    os.makedirs(vis_path, exist_ok=True)
                    os.makedirs(mask_path, exist_ok=True)

                    base = data_val["imidx"][ii].item()
                    print("base:", base)

                    vis_save_base = os.path.join(
                        vis_path, img_name
                    )  # os.path.join(args.output, str(k) + "_" + str(base))
                    mask_save_base = os.path.join(mask_path, img_name)

                    imgs_ii = imgs[ii].astype(dtype=np.uint8)
                    show_iou = torch.tensor([iou.item()])
                    show_boundary_iou = torch.tensor([boundary_iou.item()])
                    if args.prompt_type != "P" and args.prompt_type != "B":
                        masks_pa_vis[ii] = mask_opt(masks_pa_vis[ii])

                    save_timecost(img_name, timecost, timecost_path)
                    save_mask(masks_pa_vis[ii], mask_save_base)
                    show_anns(
                        masks_pa_vis[ii],
                        None,
                        labels_box[ii].cpu(),
                        None,
                        vis_save_base,
                        imgs_ii,
                        show_iou,
                        show_boundary_iou,
                    )

                    if args.token_visualisation:

                        imname = data_val["ori_im_path"][ii].split("/")[-1]
                        imfolder = (
                            data_val["ori_im_path"][ii].split("/")[-2].split("_")[0]
                        )

                        # 保存 dense 和 spare
                        dense_masks_base = os.path.join(
                            mask_path, imname.split(".")[0] + "_dense"
                        )
                        file_name_temp = dense_masks_base + ".png"
                        image_array = np.array(dense_vis[ii][0])
                        image_temp = Image.fromarray(image_array)
                        image_temp.save(file_name_temp)

                        spare_masks_base = os.path.join(
                            mask_path, imname.split(".")[0] + "_spare"
                        )
                        file_name_temp = spare_masks_base + ".png"
                        image_array = np.array(spare_vis[ii][0])
                        image_temp = Image.fromarray(image_array)
                        image_temp.save(file_name_temp)

                        # 保存 refin-map
                        ref_map_base = os.path.join(
                            mask_path, imname.split(".")[0] + "_refmap"
                        )
                        file_name_temp = ref_map_base + ".png"
                        image_array = np.array(ref_map_vis[ii][0])
                        image_temp = Image.fromarray(image_array)
                        image_temp.save(file_name_temp)

                        # 保存 coare_masks 和 refined_masks
                        refined_masks_base = os.path.join(
                            mask_path, imname.split(".")[0] + "_refined"
                        )
                        file_name_temp = refined_masks_base + ".png"
                        image_array = np.array(refined_masks_vis[ii][0])
                        image_temp = Image.fromarray(image_array)
                        image_temp.save(file_name_temp)

                        coarse_masks_base = os.path.join(
                            mask_path, imname.split(".")[0] + "_coarse"
                        )
                        file_name_temp = coarse_masks_base + ".png"
                        image_array = np.array(coarse_masks_vis[ii][0])
                        image_temp = Image.fromarray(image_array)
                        image_temp.save(file_name_temp)

                        # 保存 Uncertain
                        uncertain_masks_base = os.path.join(
                            mask_path, imname.split(".")[0] + "_uncertain"
                        )
                        file_name_temp = uncertain_masks_base + ".png"
                        image_array = np.array(uncertain_maps_vis[ii][0])
                        image_temp = Image.fromarray(image_array)
                        image_temp.save(file_name_temp)

                        # 正反例点
                        ps_base = os.path.join(mask_path, imname.split(".")[0] + "_ps")
                        file_name_temp = ps_base + ".png"
                        image_array = np.array(ps_vis[ii])
                        image_temp = Image.fromarray(image_array)
                        image_temp.save(file_name_temp)

                        ns_base = os.path.join(mask_path, imname.split(".")[0] + "_ns")
                        file_name_temp = ns_base + ".png"
                        image_array = np.array(ns_vis[ii])
                        image_temp = Image.fromarray(image_array)
                        image_temp.save(file_name_temp)

            loss_dict = {
                "val_iou_" + str(k): iou,
                "val_boundary_iou_" + str(k): boundary_iou,
            }
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)

        print("============================")
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        resstat = {
            k: meter.global_avg
            for k, meter in metric_logger.meters.items()
            if meter.count > 0
        }
        test_stats.update(resstat)

    return test_stats


if __name__ == "__main__":

    args = get_args_parser()

    ### --------------- Configuring the Train and Valid datasets ---------------
    input_dir = args.input

    HAM10000_root_folder = os.path.join(input_dir, "HAM10000/input")
    HAM10000_image_folder_train = os.path.join(
        HAM10000_root_folder, "train/HAM10000_img"
    )
    HAM10000_seg_folder_train = os.path.join(HAM10000_root_folder, "train/HAM10000_seg")
    HAM10000_p_s_folder_train = os.path.join(
        HAM10000_root_folder,
        "train/HAM10000_train_positive_"
        + args.scribble_positive_type
        + "_"
        + str(args.labeller),
    )
    HAM10000_n_s_folder_train = os.path.join(
        HAM10000_root_folder,
        "train/HAM10000_train_negative_"
        + args.scribble_negative_type
        + "_"
        + str(args.labeller),
    )

    # if args.scribblee_type is None:
    #     HAM10000_train = {
    #         "name": "HAM10000_train",
    #         "im_dir": str(HAM10000_image_folder_train),
    #         "gt_dir": str(HAM10000_seg_folder_train),
    #         "im_ext": ".jpg",
    #         "gt_ext": "_segmentation.png",
    #     }
    # else:
    HAM10000_train = {
        "name": "HAM10000_train",
        "im_dir": str(HAM10000_image_folder_train),
        "gt_dir": str(HAM10000_seg_folder_train),
        "p_scribble": str(HAM10000_p_s_folder_train),
        "n_scribble": str(HAM10000_n_s_folder_train),
        "im_ext": ".jpg",
        "gt_ext": "_segmentation.png",
    }

    HAM10000_image_folder_val = os.path.join(HAM10000_root_folder, "val/HAM10000_img")
    HAM10000_seg_folder_val = os.path.join(HAM10000_root_folder, "val/HAM10000_seg")
    HAM10000_p_s_folder_val = os.path.join(
        HAM10000_root_folder,
        "val/HAM10000_val_positive_"
        + args.scribble_positive_type
        + "_"
        + str(args.labeller),
    )
    HAM10000_n_s_folder_val = os.path.join(
        HAM10000_root_folder,
        "val/HAM10000_val_negative_"
        + args.scribble_negative_type
        + "_"
        + str(args.labeller),
    )
    HAM10000_val = {
        "name": "HAM10000_val",
        "im_dir": str(HAM10000_image_folder_val),
        "gt_dir": str(HAM10000_seg_folder_val),
        "p_scribble": str(HAM10000_p_s_folder_val),
        "n_scribble": str(HAM10000_n_s_folder_val),
        "im_ext": ".jpg",
        "gt_ext": "_segmentation.png",
    }

    HAM10000_image_folder_test = os.path.join(HAM10000_root_folder, "test/HAM10000_img")
    HAM10000_seg_folder_test = os.path.join(HAM10000_root_folder, "test/HAM10000_seg")
    HAM10000_p_s_folder_test = os.path.join(
        HAM10000_root_folder,
        "test/HAM10000_test_positive_"
        + args.scribble_positive_type
        + "_"
        + str(args.labeller),
    )
    HAM10000_n_s_folder_test = os.path.join(
        HAM10000_root_folder,
        "test/HAM10000_test_negative_"
        + args.scribble_negative_type
        + "_"
        + str(args.labeller),
    )
    HAM10000_test = {
        "name": "HAM10000_test",
        "im_dir": str(HAM10000_image_folder_test),
        "gt_dir": str(HAM10000_seg_folder_test),
        "p_scribble": str(HAM10000_p_s_folder_test),
        "n_scribble": str(HAM10000_n_s_folder_test),
        "im_ext": ".jpg",
        "gt_ext": "_segmentation.png",
    }

    ISIC_2016_root_folder = os.path.join(input_dir, "ISIC2016/input")
    ISIC_2016_image_folder = os.path.join(ISIC_2016_root_folder, "test/ISIC2016_img")
    ISIC_2016_seg_folder = os.path.join(ISIC_2016_root_folder, "test/ISIC2016_seg")
    ISIC_2016_p_s_folder_test = os.path.join(
        ISIC_2016_root_folder,
        "test/ISIC2016_positive_"
        + args.scribble_positive_type
        + "_"
        + str(args.labeller),
    )
    ISIC_2016_n_s_folder_test = os.path.join(
        ISIC_2016_root_folder,
        "test/ISIC2016_negative_"
        + args.scribble_negative_type
        + "_"
        + str(args.labeller),
    )
    ISIC_2016 = {
        "name": "ISIC_2016_test",
        "im_dir": str(ISIC_2016_image_folder),
        "gt_dir": str(ISIC_2016_seg_folder),
        "p_scribble": str(ISIC_2016_p_s_folder_test),
        "n_scribble": str(ISIC_2016_n_s_folder_test),
        "im_ext": ".jpg",
        "gt_ext": "_segmentation.png",
    }

    ISIC_2017_root_folder = os.path.join(input_dir, "ISIC2017/input")
    ISIC_2017_image_folder = os.path.join(ISIC_2017_root_folder, "test/ISIC2017_img")
    ISIC_2017_seg_folder = os.path.join(ISIC_2017_root_folder, "test/ISIC2017_seg")
    ISIC_2017_p_s_folder_test = os.path.join(
        ISIC_2017_root_folder,
        "test/ISIC2017_positive_"
        + args.scribble_positive_type
        + "_"
        + str(args.labeller),
    )
    ISIC_2017_n_s_folder_test = os.path.join(
        ISIC_2017_root_folder,
        "test/ISIC2017_negative_"
        + args.scribble_negative_type
        + "_"
        + str(args.labeller),
    )
    ISIC_2017 = {
        "name": "ISIC_2017_test",
        "im_dir": str(ISIC_2017_image_folder),
        "gt_dir": str(ISIC_2017_seg_folder),
        "p_scribble": str(ISIC_2017_p_s_folder_test),
        "n_scribble": str(ISIC_2017_n_s_folder_test),
        "im_ext": ".jpg",
        "gt_ext": "_segmentation.png",
    }

    PH2_root_folder = os.path.join(input_dir, "PH2/input")
    PH2_image_folder = os.path.join(PH2_root_folder, "test/PH2_img")
    PH2_seg_folder = os.path.join(PH2_root_folder, "test/PH2_seg")
    PH2_p_s_folder_test = os.path.join(
        PH2_root_folder,
        "test/PH2_positive_" + args.scribble_positive_type + "_" + str(args.labeller),
    )
    PH2_n_s_folder_test = os.path.join(
        PH2_root_folder,
        "test/PH2_negative_" + args.scribble_negative_type + "_" + str(args.labeller),
    )
    PH2 = {
        "name": "PH2_test",
        "im_dir": str(PH2_image_folder),
        "gt_dir": str(PH2_seg_folder),
        "p_scribble": str(PH2_p_s_folder_test),
        "n_scribble": str(PH2_n_s_folder_test),
        "im_ext": ".jpg",
        "gt_ext": "_segmentation.png",
    }

    Atlas_root_folder = os.path.join(input_dir, "Atlas/input")
    Atlas_image_folder = os.path.join(Atlas_root_folder, "test/Atlas_img")
    Atlas_seg_folder = os.path.join(Atlas_root_folder, "test/Atlas_seg")
    Atlas_p_s_folder_test = os.path.join(
        Atlas_root_folder,
        "test/Atlas_positive_" + args.scribble_positive_type + "_" + str(args.labeller),
    )
    Atlas_n_s_folder_test = os.path.join(
        Atlas_root_folder,
        "test/Atlas_negative_" + args.scribble_negative_type + "_" + str(args.labeller),
    )
    Atlas = {
        "name": "Atlas_test",
        "im_dir": str(Atlas_image_folder),
        "gt_dir": str(Atlas_seg_folder),
        "p_scribble": str(Atlas_p_s_folder_test),
        "n_scribble": str(Atlas_n_s_folder_test),
        "im_ext": ".jpg",
        "gt_ext": "_segmentation.png",
    }

    AtlasZoomIn10_root_folder = os.path.join(input_dir, "AtlasZoomIn10/input")
    AtlasZoomIn10_image_folder = os.path.join(
        AtlasZoomIn10_root_folder, "test/AtlasZoomIn10_img"
    )
    AtlasZoomIn10_seg_folder = os.path.join(
        AtlasZoomIn10_root_folder, "test/AtlasZoomIn10_seg"
    )
    AtlasZoomIn10_p_s_folder_test = os.path.join(
        AtlasZoomIn10_root_folder,
        "test/AtlasZoomIn10_positive_"
        + args.scribble_positive_type
        + "_"
        + str(args.labeller),
    )
    AtlasZoomIn10_n_s_folder_test = os.path.join(
        AtlasZoomIn10_root_folder,
        "test/AtlasZoomIn10_negative_"
        + args.scribble_negative_type
        + "_"
        + str(args.labeller),
    )
    AtlasZoomIn10 = {
        "name": "AtlasZoomIn10_test",
        "im_dir": str(AtlasZoomIn10_image_folder),
        "gt_dir": str(AtlasZoomIn10_seg_folder),
        "p_scribble": str(AtlasZoomIn10_p_s_folder_test),
        "n_scribble": str(AtlasZoomIn10_n_s_folder_test),
        "im_ext": ".jpg",
        "gt_ext": "_segmentation.png",
    }

    AtlasZoomIn35_root_folder = os.path.join(input_dir, "AtlasZoomIn35/input")
    AtlasZoomIn35_image_folder = os.path.join(
        AtlasZoomIn35_root_folder, "test/AtlasZoomIn35_img"
    )
    AtlasZoomIn35_seg_folder = os.path.join(
        AtlasZoomIn35_root_folder, "test/AtlasZoomIn35_seg"
    )
    AtlasZoomIn35_p_s_folder_test = os.path.join(
        AtlasZoomIn35_root_folder,
        "test/AtlasZoomIn35_positive_"
        + args.scribble_positive_type
        + "_"
        + str(args.labeller),
    )
    AtlasZoomIn35_n_s_folder_test = os.path.join(
        AtlasZoomIn35_root_folder,
        "test/AtlasZoomIn35_negative_"
        + args.scribble_negative_type
        + "_"
        + str(args.labeller),
    )
    AtlasZoomIn35 = {
        "name": "AtlasZoomIn35_test",
        "im_dir": str(AtlasZoomIn35_image_folder),
        "gt_dir": str(AtlasZoomIn35_seg_folder),
        "p_scribble": str(AtlasZoomIn35_p_s_folder_test),
        "n_scribble": str(AtlasZoomIn35_n_s_folder_test),
        "im_ext": ".jpg",
        "gt_ext": "_segmentation.png",
    }

    Dermofit_root_folder = os.path.join(input_dir, "Dermofit/input")
    Dermofit_image_folder = os.path.join(Dermofit_root_folder, "test/Dermofit_img")
    Dermofit_seg_folder = os.path.join(Dermofit_root_folder, "test/Dermofit_seg")
    Dermofit_p_s_folder_test = os.path.join(
        Dermofit_root_folder,
        "test/Dermofit_positive_"
        + args.scribble_positive_type
        + "_"
        + str(args.labeller),
    )
    Dermofit_n_s_folder_test = os.path.join(
        Dermofit_root_folder,
        "test/Dermofit_negative_"
        + args.scribble_negative_type
        + "_"
        + str(args.labeller),
    )
    Dermofit = {
        "name": "Dermofit_test",
        "im_dir": str(Dermofit_image_folder),
        "gt_dir": str(Dermofit_seg_folder),
        "p_scribble": str(Dermofit_p_s_folder_test),
        "n_scribble": str(Dermofit_n_s_folder_test),
        "im_ext": ".jpg",
        "gt_ext": "_segmentation.png",
    }

    if args.eval:
        train_datasets = None
        valid_datasets = [
            HAM10000_val,
            HAM10000_test,
            ISIC_2016,
            ISIC_2017,
            PH2,
            # Atlas,
            AtlasZoomIn10,  # padding 10
            # AtlasZoomIn35,  # padding 35
            Dermofit,
        ]
    else:
        train_datasets = [HAM10000_train]
        valid_datasets = [HAM10000_val]

    net = MaskDecoderPA(args.model_type)

    if not args.eval:
        wandb.init(
            project="PSAM",
            tags=["baseline", "labeller=" + str(args.labeller)],
            name="resize_" + str(args.input_size[0]),
            notes=args.comments,
        )
        config = wandb.config
        wandb.watch(net, log="all")
        config.args = args

    main(net, train_datasets, valid_datasets, args)
