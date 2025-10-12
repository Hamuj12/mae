# tools/make_space_probe_split.py
import os, random, shutil, pathlib
from argparse import ArgumentParser

def main():
    ap = ArgumentParser()
    ap.add_argument("--src", required=True, help="root with class folders")
    ap.add_argument("--dst", required=True, help="probe root to create")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--per_class_cap", type=int, default=400, help="0 disables")
    ap.add_argument("--exts", default=".jpg,.jpeg,.png")
    args = ap.parse_args()

    exts = tuple(x.strip().lower() for x in args.exts.split(","))
    classes = [d for d in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, d))]
    for split in ["train", "val"]:
        for c in classes:
            os.makedirs(os.path.join(args.dst, split, c), exist_ok=True)

    for c in classes:
        src_dir = os.path.join(args.src, c)
        imgs = [str(p) for p in pathlib.Path(src_dir).rglob("*") if p.suffix.lower() in exts]
        random.shuffle(imgs)
        if args.per_class_cap > 0:
            imgs = imgs[:args.per_class_cap]
        n_val = max(1, int(len(imgs) * args.val_frac))
        val_imgs = imgs[:n_val]
        train_imgs = imgs[n_val:]
        for s, lst in [("train", train_imgs), ("val", val_imgs)]:
            dst_dir = os.path.join(args.dst, s, c)
            for p in lst:
                shutil.copy2(p, os.path.join(dst_dir, os.path.basename(p)))
    print("Done. Probe at:", args.dst)

if __name__ == "__main__":
    main()