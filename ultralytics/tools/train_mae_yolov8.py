import argparse

from ultralytics import YOLO


def main(opt):
    model = YOLO(opt.model)
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
    main(parser.parse_args())
