import argparse

from ultralytics import YOLO


def _orbitgen_loader(data_path, img_size, batch_size):
    """Create a DataLoader for OrbitGen datasets."""
    from torch.utils.data import DataLoader

    from ultralytics.nn.datasets.orbitgen_dataset import OrbitGenDataset
    from ultralytics.nn.utils.collate import orbitgen_collate_fn

    dataset = OrbitGenDataset(data_path, img_size=img_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=orbitgen_collate_fn)


def main(opt):
    model = YOLO(opt.model)
    if opt.orbitgen:
        train_loader = _orbitgen_loader(opt.data, opt.img, opt.batch)
        model.train(dataloader=train_loader, imgsz=opt.img, epochs=opt.epochs, device=opt.device)
    else:
        model.train(
            data=opt.data,
            imgsz=opt.img,
            epochs=opt.epochs,
            batch=opt.batch,
            device=opt.device,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model YAML path")
    parser.add_argument("--data", type=str, required=True, help="Data YAML path")
    parser.add_argument("--img", type=int, default=640, help="Image size")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--orbitgen", action="store_true", help="Use OrbitGen dataset")
    main(parser.parse_args())
