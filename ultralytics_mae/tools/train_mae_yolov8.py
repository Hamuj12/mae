"""Training entry point for MAE-initialized YOLO models saved as .pt checkpoints."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main(opt: argparse.Namespace) -> None:
    model_path = Path(opt.model)
    if model_path.suffix != ".pt":
        raise ValueError("--model must point to a .pt checkpoint built with build_mae_yolo.py")

    model = YOLO(model_path)
    train_kwargs = dict(
        data=opt.data,
        imgsz=opt.img,
        epochs=opt.epochs,
        batch=opt.batch,
        device=opt.device,
    )

    if opt.orbitgen:
        print("[INFO] --orbitgen flag detected; using standard Ultralytics training with generated labels.")

    model.train(**train_kwargs)


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
