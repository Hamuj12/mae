# tools/clip_label_space.py
import argparse, os, torch, torchvision.transforms as T
from PIL import Image, ImageStat
from pathlib import Path
from tqdm import tqdm
import shutil
import clip

PROMPTS = {
    "earth_limb": [
        "a curved blue atmosphere seen from orbit",
        "the earth's limb viewed from space",
        "planetary horizon with a thin blue atmosphere",
    ],
    "deep_space": [
        "a dark sky filled with many point-like stars",
        "a star field in deep space at night",
        "black space with numerous small stars",
    ],
    "deepfield": [
        "a deep space image with many distant galaxies",
        "a telescope deep field with colorful galaxies",
        "numerous faint galaxies in a cosmic deep field",
    ],
    # "pure_black" handled by heuristic; no prompts needed
}

def is_pure_black(pil_img, v_thresh=10, std_thresh=5):
    """
    Quick heuristic for near-empty frames.
    Args use 0..255 scale for mean/std.
    """
    g = pil_img.convert("L")
    stat = ImageStat.Stat(g)
    mean = stat.mean[0]
    # Pillow's ImageStat on some builds exposes var only; stddev on others.
    std = getattr(stat, "stddev", [None])[0]
    if std is None:
        # fallback from variance
        var = getattr(stat, "var", [1e9])[0]
        std = (var ** 0.5)
    return (mean < v_thresh) and (std < std_thresh)

def build_text_emb(model, device):
    texts, cls_names = [], []
    for cls, prompts in PROMPTS.items():
        for p in prompts:
            texts.append(p)
            cls_names.append(cls)
    tok = clip.tokenize(texts).to(device)
    with torch.no_grad():
        txt = model.encode_text(tok)
        txt = txt / txt.norm(dim=-1, keepdim=True)
    return txt, cls_names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder of images")
    ap.add_argument("--dst", required=True, help="output root (ImageFolder)")
    ap.add_argument("--exts", default=".jpg,.jpeg,.png", type=str)
    ap.add_argument("--model", default="ViT-B/32")
    ap.add_argument("--threshold", type=float, default=0.23, help="min prob to accept a class")
    ap.add_argument("--margin", type=float, default=0.06, help="min top1-top2 prob margin")
    ap.add_argument("--max_per_class", type=int, default=400, help="cap per class (<=0 disables)")
    ap.add_argument("--pure_black_mean", type=int, default=10, help="0..255 mean cutoff for pure black")
    ap.add_argument("--pure_black_std", type=int, default=5, help="0..255 std cutoff for pure black")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.model, device=device)
    model.eval()
    text_emb, class_names = build_text_emb(model, device)
    exts = tuple(x.strip().lower() for x in args.exts.split(","))

    classes = list(PROMPTS.keys()) + ["pure_black"]
    counts = {c: 0 for c in classes}
    for c in classes:
        os.makedirs(os.path.join(args.dst, c), exist_ok=True)

    paths = [p for p in Path(args.src).rglob("*") if p.suffix.lower() in exts]
    softmax = torch.nn.Softmax(dim=-1)

    for p in tqdm(paths):
        # 1) open PIL first so we can run the heuristic
        try:
            pil = Image.open(p).convert("RGB")
        except Exception:
            continue

        # 2) pure_black heuristic (skip CLIP if true)
        if is_pure_black(pil, args.pure_black_mean, args.pure_black_std):
            if args.max_per_class <= 0 or counts["pure_black"] < args.max_per_class:
                try:
                    shutil.copy2(str(p), os.path.join(args.dst, "pure_black", p.name))
                    counts["pure_black"] += 1
                except Exception:
                    pass
            continue

        # 3) CLIP inference
        img = preprocess(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            img_emb = model.encode_image(img)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            logits = 100.0 * img_emb @ text_emb.T
            probs = softmax(logits).squeeze(0).tolist()

        # 4) aggregate prompt-scores -> class-scores via max
        cls_scores = {}
        for idx, cls in enumerate(class_names):
            cls_scores[cls] = max(cls_scores.get(cls, 0.0), probs[idx])

        # 5) accept only if confident (threshold + margin over 2nd best)
        ranked = sorted(cls_scores.items(), key=lambda kv: kv[1], reverse=True)
        best_cls, best_prob = ranked[0]
        second_prob = ranked[1][1] if len(ranked) > 1 else 0.0

        if best_prob >= args.threshold and (best_prob - second_prob) >= args.margin:
            if args.max_per_class <= 0 or counts[best_cls] < args.max_per_class:
                dst_path = os.path.join(args.dst, best_cls, p.name)
                try:
                    shutil.copy2(str(p), dst_path)
                    counts[best_cls] += 1
                except Exception:
                    pass
        # else: skip to keep the probe clean

    print("Class counts:", counts)

if __name__ == "__main__":
    main()