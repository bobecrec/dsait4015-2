import json
import os
import csv
import time
import numpy as np
import torch
from pathlib import Path
from torchvision.utils import save_image
from torchvision.models import vgg16
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent


# -----------------------------
# Utility: parse ImageNet prediction (top-1)
# -----------------------------
def parse_prediction(output, categories):
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probs, 1)
    return categories[int(top_catid.item())], float(top_prob.item())


def topk_predictions(output, categories, k=5):
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probs, k)
    return [
        {"idx": int(i.item()), "label": categories[int(i.item())], "prob": float(p.item())}
        for p, i in zip(top_prob, top_catid)
    ]


def compute_perturbation_metrics(x, x_adv, pixel_change_threshold=1.0 / 255.0):
    """
    x, x_adv: torch tensors of shape [1, C, H, W] in [0,1]
    Returns: (linf, l2, num_pixels_changed)
    """
    delta = (x_adv - x).detach()
    abs_delta = delta.abs()

    linf = float(abs_delta.max().item())
    l2 = float(torch.norm(delta.view(-1), p=2).item())

    # count pixels (H*W) where ANY channel exceeds threshold
    per_pixel_changed = (abs_delta > pixel_change_threshold).any(dim=1)  # [1,H,W]
    num_pixels_changed = int(per_pixel_changed.sum().item())

    return linf, l2, num_pixels_changed


def safe_label_to_index(label_to_index, label):
    return int(label_to_index[label]) if label in label_to_index else -1


# ================================================================
# 1. Load JSON file with images + expected human label
# ================================================================
JSON_FILE = "data/image_labels.json"
IMAGE_DIR = "images/"

with open(JSON_FILE, "r") as f:
    items = json.load(f)

# ================================================================
# 2. Load ImageNet labels
# ================================================================
with open("data/imagenet_classes.txt", "r") as f:
    imagenet_labels = [s.strip() for s in f.readlines()]

label_to_index = {label: i for i, label in enumerate(imagenet_labels)}

# ================================================================
# 3. Model
# ================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
net = vgg16(weights="DEFAULT").to(device)
net.eval()

# ================================================================
# 4. Image preprocessing transform
# ================================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ================================================================
# 5. Attack hyperparameters
# ================================================================
EPS = 0.30
PGD_STEPS = 40
PGD_STEP_SIZE = 0.01

# ================================================================
# 6. Output directory + logging
# ================================================================
OUTDIR = Path("attack_results")
IMG_OUTDIR = OUTDIR / "images"
OUTDIR.mkdir(parents=True, exist_ok=True)
IMG_OUTDIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTDIR / "attack_stats.csv"
JSONL_PATH = OUTDIR / "attack_stats.jsonl"

fieldnames = [
    "image", "human_label", "true_idx",
    "clean_top1_label", "clean_top1_idx", "clean_top1_prob",

    "fgm_top1_label", "fgm_top1_idx", "fgm_top1_prob",
    "fgm_success", "fgm_changed_pred",
    "fgm_linf", "fgm_l2", "fgm_num_pixels_changed",
    "fgm_runtime_s",

    "pgd_top1_label", "pgd_top1_idx", "pgd_top1_prob",
    "pgd_success", "pgd_changed_pred",
    "pgd_linf", "pgd_l2", "pgd_num_pixels_changed",
    "pgd_runtime_s",
]

csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8")
writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
writer.writeheader()

jsonl_file = open(JSONL_PATH, "w", encoding="utf-8")

print(f"Writing CSV to:   {CSV_PATH}")
print(f"Writing JSONL to: {JSONL_PATH}")

# ================================================================
# 7. Run attacks for every image
# ================================================================
for entry in tqdm(items, desc="Running attacks"):
    image_file = entry["image"]
    human_label = entry["label"]

    # -----------------------------
    # Load + preprocess image
    # -----------------------------
    img_path = os.path.join(IMAGE_DIR, image_file)
    img_pil = Image.open(img_path).convert("RGB")
    x = transform(img_pil).unsqueeze(0).to(device)

    # -----------------------------
    # Ground truth index (from human label)
    # -----------------------------
    true_idx = safe_label_to_index(label_to_index, human_label)
    if true_idx == -1:
        print(f"⚠️ Warning: '{human_label}' not found in ImageNet labels.")

    # -----------------------------
    # Predict clean image
    # -----------------------------
    with torch.no_grad():
        out_clean = net(x)

    clean_top1_label, clean_top1_prob = parse_prediction(out_clean, imagenet_labels)
    clean_top1_idx = safe_label_to_index(label_to_index, clean_top1_label)
    clean_top5 = topk_predictions(out_clean, imagenet_labels, k=5)

    clean_path = IMG_OUTDIR / f"{image_file}_clean.png"
    save_image(x, str(clean_path))

    # =====================================================
    # FGM Attack (timed)
    # =====================================================
    t0 = time.perf_counter()
    x_fgm = fast_gradient_method(net, x, EPS, np.inf)
    fgm_runtime = time.perf_counter() - t0

    with torch.no_grad():
        out_fgm = net(x_fgm)

    fgm_top1_label, fgm_top1_prob = parse_prediction(out_fgm, imagenet_labels)
    fgm_top1_idx = safe_label_to_index(label_to_index, fgm_top1_label)
    fgm_top5 = topk_predictions(out_fgm, imagenet_labels, k=5)

    fgm_path = IMG_OUTDIR / f"{image_file}_fgm.png"
    save_image(x_fgm, str(fgm_path))

    fgm_linf, fgm_l2, fgm_num_pix = compute_perturbation_metrics(x, x_fgm)
    fgm_changed_pred = (
        int(fgm_top1_idx != clean_top1_idx) if (clean_top1_idx != -1 and fgm_top1_idx != -1) else -1
    )
    fgm_success = (
        int(fgm_top1_idx != true_idx) if (true_idx != -1 and fgm_top1_idx != -1) else -1
    )

    # =====================================================
    # PGD Attack (timed)
    # =====================================================
    t0 = time.perf_counter()
    x_pgd = projected_gradient_descent(net, x, EPS, PGD_STEP_SIZE, PGD_STEPS, np.inf)
    pgd_runtime = time.perf_counter() - t0

    with torch.no_grad():
        out_pgd = net(x_pgd)

    pgd_top1_label, pgd_top1_prob = parse_prediction(out_pgd, imagenet_labels)
    pgd_top1_idx = safe_label_to_index(label_to_index, pgd_top1_label)
    pgd_top5 = topk_predictions(out_pgd, imagenet_labels, k=5)

    pgd_path = IMG_OUTDIR / f"{image_file}_pgd.png"
    save_image(x_pgd, str(pgd_path))

    pgd_linf, pgd_l2, pgd_num_pix = compute_perturbation_metrics(x, x_pgd)
    pgd_changed_pred = (
        int(pgd_top1_idx != clean_top1_idx) if (clean_top1_idx != -1 and pgd_top1_idx != -1) else -1
    )
    pgd_success = (
        int(pgd_top1_idx != true_idx) if (true_idx != -1 and pgd_top1_idx != -1) else -1
    )

    # -----------------------------
    # Write CSV row
    # -----------------------------
    row = {
        "image": image_file,
        "human_label": human_label,
        "true_idx": true_idx,

        "clean_top1_label": clean_top1_label,
        "clean_top1_idx": clean_top1_idx,
        "clean_top1_prob": clean_top1_prob,

        "fgm_top1_label": fgm_top1_label,
        "fgm_top1_idx": fgm_top1_idx,
        "fgm_top1_prob": fgm_top1_prob,
        "fgm_success": fgm_success,
        "fgm_changed_pred": fgm_changed_pred,
        "fgm_linf": fgm_linf,
        "fgm_l2": fgm_l2,
        "fgm_num_pixels_changed": fgm_num_pix,
        "fgm_runtime_s": fgm_runtime,

        "pgd_top1_label": pgd_top1_label,
        "pgd_top1_idx": pgd_top1_idx,
        "pgd_top1_prob": pgd_top1_prob,
        "pgd_success": pgd_success,
        "pgd_changed_pred": pgd_changed_pred,
        "pgd_linf": pgd_linf,
        "pgd_l2": pgd_l2,
        "pgd_num_pixels_changed": pgd_num_pix,
        "pgd_runtime_s": pgd_runtime,
    }
    writer.writerow(row)
    csv_file.flush()

    # -----------------------------
    # Write JSONL record
    # -----------------------------
    record = {
        "image": image_file,
        "human_label": human_label,
        "true_idx": true_idx,
        "paths": {
            "clean": str(clean_path),
            "fgm": str(fgm_path),
            "pgd": str(pgd_path),
        },
        "clean": {
            "top1": {"idx": clean_top1_idx, "label": clean_top1_label, "prob": clean_top1_prob},
            "top5": clean_top5,
        },
        "fgm": {
            "eps": EPS,
            "runtime_s": fgm_runtime,
            "top1": {"idx": fgm_top1_idx, "label": fgm_top1_label, "prob": fgm_top1_prob},
            "top5": fgm_top5,
            "success": fgm_success,
            "changed_pred": fgm_changed_pred,
            "perturbation": {"linf": fgm_linf, "l2": fgm_l2, "num_pixels_changed": fgm_num_pix},
        },
        "pgd": {
            "eps": EPS,
            "steps": PGD_STEPS,
            "step_size": PGD_STEP_SIZE,
            "runtime_s": pgd_runtime,
            "top1": {"idx": pgd_top1_idx, "label": pgd_top1_label, "prob": pgd_top1_prob},
            "top5": pgd_top5,
            "success": pgd_success,
            "changed_pred": pgd_changed_pred,
            "perturbation": {"linf": pgd_linf, "l2": pgd_l2, "num_pixels_changed": pgd_num_pix},
        },
    }
    jsonl_file.write(json.dumps(record) + "\n")
    jsonl_file.flush()

csv_file.close()
jsonl_file.close()
print("\nDone.")
print(f"Saved CSV:   {CSV_PATH}")
print(f"Saved JSONL: {JSONL_PATH}")
