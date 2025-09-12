import argparse
from collections import OrderedDict

import torch


def main(opt):
    ckpt = torch.load(opt.ckpt, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    encoder = OrderedDict()
    prefix = "model.model.encoder."
    for k, v in state_dict.items():
        if k.startswith(prefix):
            encoder[k[len(prefix):]] = v
    torch.save(encoder, opt.out)
    print(f"Saved encoder weights to {opt.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Lightning .ckpt path")
    parser.add_argument("--out", type=str, required=True, help="Output .pt path")
    main(parser.parse_args())
