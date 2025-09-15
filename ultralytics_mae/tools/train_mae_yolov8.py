"""Training entry point for MAE-initialized YOLO models saved as .pt checkpoints."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def _orbitgen_loader(data_yaml, img_size, batch_size):
    """Create DataLoaders for OrbitGen datasets using train/val/test splits."""
    import yaml
    from torch.utils.data import DataLoader
    from ultralytics_mae.nn.datasets.orbitgen_dataset import OrbitGenDataset
    from ultralytics_mae.nn.utils.collate import orbitgen_collate_fn

    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["path"])
    loaders = {}

    for split in ["train", "val", "test"]:
        if split in cfg and cfg[split]:
            dataset = OrbitGenDataset(root, split=split, img_size=img_size)
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                collate_fn=orbitgen_collate_fn,
            )

    return loaders


def main(opt: argparse.Namespace) -> None:
    model_path = Path(opt.model)
    if model_path.suffix != ".pt":
        raise ValueError("--model must point to a .pt checkpoint built with build_mae_yolo.py")

    model = YOLO(model_path)
    if opt.orbitgen:
        train_loader = _orbitgen_loader(opt.data, opt.img, opt.batch)
        val_loader = _orbitgen_loader(opt.data, opt.img, opt.batch)
        test_loader = _orbitgen_loader(opt.data, opt.img, opt.batch)

        # YOLO.train only takes train/val dataloaders; test is usually separate
        model.train(
            dataloader=train_loader,
            val_dataloader=val_loader,
            imgsz=opt.img,
            epochs=opt.epochs,
            device=opt.device,
        )
        # Optionally evaluate on test set
        if len(test_loader.dataset) > 0:
            model.val(dataloader=test_loader, imgsz=opt.img, device=opt.device)
    else:
        model.train(
            data=opt.data,
            imgsz=opt.img,
            epochs=opt.epochs,
            batch=opt.batch,
            device=opt.device,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True, help="Path to MAE+YOLO .pt checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Dataset YAML path")
    parser.add_argument("--img", type=int, default=640, help="Image size")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--orbitgen", action="store_true", help="Use OrbitGen dataset pipeline")
    main(parser.parse_args())