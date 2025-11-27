import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler

from utils import *

# from .atlas import Atlas
from .brat import Brat
from .ddti import DDTI

# from .isic import ISIC2016
from .kits import KITS
from .lidc import LIDC
from .lnq import LNQ
from .pendal import Pendal
from .refuge import REFUGE
from .segrap import SegRap
from .stare import STARE
from .toothfairy import ToothFairy
from .wbc import WBC
from .dataset import HAM10000, ISIC2016, ISIC2017, PH2, AtlasZoomIn10, Dermofit


def get_dataloader(args):
    transform_train = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )

    transform_train_seg = transforms.Compose(
        [
            transforms.Resize((args.out_size, args.out_size)),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )

    transform_test_seg = transforms.Compose(
        [
            transforms.Resize((args.out_size, args.out_size)),
            transforms.ToTensor(),
        ]
    )

    if args.dataset == "HAM10000":
        train_dataset = HAM10000(
            args,
            args.data_path,
            transform=transform_train,
            transform_msk=transform_train_seg,
            mode="train",
        )
        test_dataset = HAM10000(
            args,
            args.data_path,
            transform=transform_test,
            transform_msk=transform_test_seg,
            mode="test",
        )

        nice_train_loader = DataLoader(
            train_dataset,
            batch_size=args.b,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        nice_test_loader = DataLoader(
            test_dataset,
            batch_size=args.b,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
    elif args.dataset == "HAM10000_val":
        train_dataset = HAM10000(
            args,
            args.data_path,
            transform=transform_train,
            transform_msk=transform_train_seg,
            mode="val",
        )
        test_dataset = HAM10000(
            args,
            args.data_path,
            transform=transform_test,
            transform_msk=transform_test_seg,
            mode="val",
        )

        nice_train_loader = DataLoader(
            train_dataset,
            batch_size=args.b,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        nice_test_loader = DataLoader(
            test_dataset,
            batch_size=args.b,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
    elif args.dataset == "HAM10000_test":
        train_dataset = HAM10000(
            args,
            args.data_path,
            transform=transform_train,
            transform_msk=transform_train_seg,
            mode="test",
        )
        test_dataset = HAM10000(
            args,
            args.data_path,
            transform=transform_test,
            transform_msk=transform_test_seg,
            mode="test",
        )
        nice_train_loader = DataLoader(
            train_dataset,
            batch_size=args.b,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        nice_test_loader = DataLoader(
            test_dataset,
            batch_size=args.b,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
    elif args.dataset == "ISIC2016_test":
        train_dataset = ISIC2016(
            args,
            args.data_path,
            transform=transform_train,
            transform_msk=transform_train_seg,
            mode="test",
        )
        test_dataset = ISIC2016(
            args,
            args.data_path,
            transform=transform_test,
            transform_msk=transform_test_seg,
            mode="test",
        )

        nice_train_loader = DataLoader(
            train_dataset,
            batch_size=args.b,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        nice_test_loader = DataLoader(
            test_dataset,
            batch_size=args.b,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
    elif args.dataset == "ISIC2017_test":
        train_dataset = ISIC2017(
            args,
            args.data_path,
            transform=transform_train,
            transform_msk=transform_train_seg,
            mode="test",
        )
        test_dataset = ISIC2017(
            args,
            args.data_path,
            transform=transform_test,
            transform_msk=transform_test_seg,
            mode="test",
        )

        nice_train_loader = DataLoader(
            train_dataset,
            batch_size=args.b,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        nice_test_loader = DataLoader(
            test_dataset,
            batch_size=args.b,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
    elif args.dataset == "PH2_test":
        train_dataset = PH2(
            args,
            args.data_path,
            transform=transform_train,
            transform_msk=transform_train_seg,
            mode="test",
        )
        test_dataset = PH2(
            args,
            args.data_path,
            transform=transform_test,
            transform_msk=transform_test_seg,
            mode="test",
        )

        nice_train_loader = DataLoader(
            train_dataset,
            batch_size=args.b,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        nice_test_loader = DataLoader(
            test_dataset,
            batch_size=args.b,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
    elif args.dataset == "AtlasZoomIn10_test":
        train_dataset = AtlasZoomIn10(
            args,
            args.data_path,
            transform=transform_train,
            transform_msk=transform_train_seg,
            mode="test",
        )
        test_dataset = AtlasZoomIn10(
            args,
            args.data_path,
            transform=transform_test,
            transform_msk=transform_test_seg,
            mode="test",
        )

        nice_train_loader = DataLoader(
            train_dataset,
            batch_size=args.b,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        nice_test_loader = DataLoader(
            test_dataset,
            batch_size=args.b,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
    elif args.dataset == "Dermofit_test":
        train_dataset = Dermofit(
            args,
            args.data_path,
            transform=transform_train,
            transform_msk=transform_train_seg,
            mode="test",
        )
        test_dataset = Dermofit(
            args,
            args.data_path,
            transform=transform_test,
            transform_msk=transform_test_seg,
            mode="test",
        )

        nice_train_loader = DataLoader(
            train_dataset,
            batch_size=args.b,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        nice_test_loader = DataLoader(
            test_dataset,
            batch_size=args.b,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
    else:
        print("the dataset is not supported now!!!")

    return nice_train_loader, nice_test_loader
