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

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args, transform=None):
    """Build dataset for training or validation.

    If standard ImageNet-style ``train`` and ``val`` folders exist, they are
    used directly. Otherwise a random split of the data at ``args.data_path`` is
    created using ``args.val_split``.
    """
    if transform is None:
        transform = build_transform(is_train, args)

    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        root = train_dir if is_train else val_dir
        dataset = datasets.ImageFolder(root, transform=transform)
    else:
        full_dataset = datasets.ImageFolder(args.data_path, transform=transform)
        if not hasattr(args, 'train_indices'):
            generator = torch.Generator().manual_seed(getattr(args, 'seed', 0))
            indices = torch.randperm(len(full_dataset), generator=generator)
            val_len = int(len(full_dataset) * args.val_split)
            args.val_indices = indices[:val_len].tolist()
            args.train_indices = indices[val_len:].tolist()
        indices = args.train_indices if is_train else args.val_indices
        dataset = Subset(full_dataset, indices)

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
