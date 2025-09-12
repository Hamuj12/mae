# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import torch

from torch.utils.data import Subset
from torchvision import datasets, transforms
from torch.utils.data import random_split

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def build_dataset(is_train, args):
    """Create a dataset for training or validation.

    If ``data_path`` contains ``train`` and ``val`` subdirectories, the
    corresponding ImageFolder is returned. Otherwise the images in
    ``data_path`` are deterministically split into training and validation
    subsets using ``args.val_split`` (a fraction between 0 and 1) and
    ``args.seed`` for reproducibility.
    """
    transform = build_transform(is_train, args)

    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        root = train_dir if is_train else val_dir
        dataset = datasets.ImageFolder(root, transform=transform)
    else:
        if not 0 < args.val_split < 1:
            raise ValueError("val_split must be between 0 and 1")
        full_dataset = datasets.ImageFolder(args.data_path, transform=transform)
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )
        dataset = train_dataset if is_train else val_dataset
    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
