"""
Assignment 2 – Adversarial Image Attack via Hill Climbing

You MUST implement:
    - compute_fitness
    - mutate_seed
    - select_best
    - hill_climb

DO NOT change function signatures.
"""
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from keras.applications import vgg16
from keras.applications.imagenet_utils import decode_predictions
from keras.utils import array_to_img, load_img, img_to_array
from tensorflow.python.keras.backend import epsilon
from torch import initial_seed
from tqdm import tqdm

from mutations import create_patch
from mutations import find_edges
from mutations import create_noise
from mutations import create_2d_signal
from mutations import create_1d_signal
from mutations import apply_noise


# ============================================================
# 1. FITNESS FUNCTION
# ============================================================

def compute_fitness(
    image_array: np.ndarray,
    model,
    target_label: str
) -> float:
    """
    Compute fitness of an image for hill climbing.

    Fitness definition (LOWER is better):
        - If the model predicts target_label:
              fitness = probability(target_label)
        - Otherwise:
              fitness = -probability(predicted_label)
    """

    pred = model.predict(np.expand_dims(image_array, axis=0), verbose=0)
    decoded = decode_predictions(pred, top=5)

    if decoded[0][0][1] == target_label:
        return decoded[0][0][2] - decoded[0][1][2]
    else:
        return -decoded[0][0][2]

    # TODO (student)
    # raise NotImplementedError("compute_fitness must be implemented by the student.")


# ============================================================
# 2. MUTATION FUNCTION
# ============================================================

def mutate_seed(
    seed: np.ndarray,
    epsilon: float
) -> List[np.ndarray]:
    """
    Produce ANY NUMBER of mutated neighbors.

    Students may implement ANY mutation strategy:
        - modify 1 pixel
        - modify multiple pixels
        - patch-based mutation
        - channel-based mutation
        - gaussian noise (clipped)
        - etc.

    BUT EVERY neighbor must satisfy the L∞ constraint:

        For all pixels i,j,c:
            |neighbor[i,j,c] - seed[i,j,c]| <= 255 * epsilon

    Requirements:
        ✓ Return a list of neighbors: [neighbor1, neighbor2, ..., neighborK]
        ✓ K can be ANY size ≥ 1
        ✓ Neighbors must be deep copies of seed
        ✓ Pixel values must remain in [0, 255]
        ✓ Must obey the L∞ bound exactly

    Args:
        seed (np.ndarray): input image
        epsilon (float): allowed perturbation budget

    Returns:
        List[np.ndarray]: mutated neighbors
    """

    # Play around with those
    # img_1d_signal = apply_noise(seed, create_1d_signal(seed.shape, int(np.random.rand() * 10), epsilon))
    # img_2d_signal = apply_noise(seed, create_2d_signal(seed.shape, 30, int(np.random.rand() * 10), epsilon))
    # img_noisy = apply_noise(seed, create_noise(seed.shape, 30, 0.1, epsilon))
    # img_patch = apply_noise(seed, create_patch(seed.shape, int(seed.shape[0]/3), epsilon))
    # img_noisy_edges = apply_noise(seed, find_edges(seed, 100, 0.4, epsilon))

    # return [
    #     img_1d_signal,
    #     img_2d_signal,
    #     img_noisy,
    #     img_noisy_edges,
    #     img_patch
    # ]
    candidates = []
    candidates.extend([apply_noise(seed, create_noise(seed.shape, 300, 5 / (seed.shape[0]*seed.shape[1]), epsilon)) for _ in range(5)])
    # candidates.extend([apply_noise(seed, find_edges(seed, 100, 0.4, epsilon)) for _ in range(1)])
    # candidates.append(apply_noise(seed, create_2d_signal(seed.shape, 30, int(np.random.rand() * 10), epsilon)))
    return candidates




# ============================================================
# 3. SELECT BEST CANDIDATE
# ============================================================

def select_best(
    candidates: List[np.ndarray],
    model,
    target_label: str
) -> Tuple[np.ndarray, float]:
    """
    Evaluate fitness for all candidates and return the one with
    the LOWEST fitness score.

    Args:
        candidates (List[np.ndarray])
        model: classifier
        target_label (str)

    Returns:
        (best_image, best_fitness)
    """

    fitness = [compute_fitness(img, model, target_label) for img in candidates]
    lowest_idx = np.argmin(fitness)
    return candidates[lowest_idx], fitness[lowest_idx]


# ============================================================
# 4. HILL-CLIMBING ALGORITHM
# ============================================================

def hill_climb(
    initial_seed: np.ndarray,
    model,
    target_label: str,
    epsilon: float = 0.30,
    iterations: int = 300
) -> Tuple[np.ndarray, float]:
    """
    Main hill-climbing loop.

    Requirements:
        ✓ Start from initial_seed
        ✓ EACH iteration:
              - Generate ANY number of neighbors using mutate_seed()
              - Enforce the SAME L∞ bound relative to initial_seed
              - Add current image to candidates (elitism)
              - Use select_best() to pick the winner
        ✓ Accept new candidate only if fitness improves
        ✓ Stop if:
              - target class is broken confidently, OR
              - no improvement for multiple steps (optional)

    Returns:
        (final_image, final_fitness)
    """
    # TODO (team work)
    img = initial_seed
    fitness = compute_fitness(img, model, target_label)
    accepted = 0
    for i in range(iterations-1):
        # Using this as epsilon results in much better images, but it is also much slower
        # It starts with a very low value and increases it by a bit every iteration which means
        # the changes are much more subtle
        epsilon_current = epsilon*(i+1)/iterations

        proposals = mutate_seed(img, epsilon)
        img_new, _ = select_best(proposals, model, target_label)
        img_new = np.clip(
            img_new,
            initial_seed - epsilon*255,
            initial_seed + epsilon*255
        )
        fitness_new = compute_fitness(img_new, model, target_label)


        if fitness_new < 0: # we have found an image that breaks the model, return
            return img_new, fitness_new

        if fitness_new < fitness:
            print(f"Old: {fitness}, New: {fitness_new}")
            img = img_new
            fitness = fitness_new
            accepted += 1

    return img, fitness


def keras_predict_top1(model, image_array: np.ndarray):
    """
    image_array: [H,W,C] in [0,255] (float32 ok)
    Returns: (top1_label: str, top1_prob: float, top1_idx: int, decoded_top5: list)
    """
    pred = model.predict(np.expand_dims(image_array, axis=0), verbose=0)  # [1,1000]
    decoded_top5 = decode_predictions(pred, top=5)[0]  # list of tuples (synset, label, prob)

    top1_label = decoded_top5[0][1]
    top1_prob = float(decoded_top5[0][2])
    top1_idx = int(np.argmax(pred[0]))  # 0..999

    return top1_label, top1_prob, top1_idx, decoded_top5


def compute_perturbation_metrics_np(x: np.ndarray, x_adv: np.ndarray):
    """
    x, x_adv: np arrays [H,W,C] in [0,255]
    Returns: (linf, l2, num_pixels_changed, percentage_pixels_changed)

    Pixel changed = ANY channel differs (no threshold).
    """
    x_f = x.astype(np.float32)
    x_adv_f = x_adv.astype(np.float32)

    delta = x_adv_f - x_f
    abs_delta = np.abs(delta)

    linf = float(abs_delta.max())
    l2 = float(np.linalg.norm(delta.reshape(-1), ord=2))

    per_pixel_changed = (abs_delta > 0).any(axis=2)  # [H,W]
    num_pixels_changed = int(per_pixel_changed.sum())

    H, W = per_pixel_changed.shape
    percentage_pixels_changed = num_pixels_changed / float(H * W)

    return linf, l2, num_pixels_changed, percentage_pixels_changed


def compute_hc_eval_metrics(
    seed_img: np.ndarray,
    adv_img: np.ndarray,
    model,
    human_label: str,
    label_to_index: dict
):
    # clean
    clean_top1_label, clean_top1_prob, clean_top1_idx, _ = keras_predict_top1(model, seed_img)

    # adv
    hc_top1_label, hc_top1_prob, hc_top1_idx, _ = keras_predict_top1(model, adv_img)

    # ground truth idx from provided imagenet_classes.txt mapping
    true_idx = label_to_index.get(human_label, -1)

    hc_changed_pred = int(hc_top1_idx != clean_top1_idx) if (clean_top1_idx != -1 and hc_top1_idx != -1) else -1
    hc_success = int(hc_top1_idx != true_idx) if (true_idx != -1 and hc_top1_idx != -1) else -1

    hc_linf, hc_l2, hc_num_pix, hc_perc = compute_perturbation_metrics_np(seed_img, adv_img)

    return {
        # clean (optional to log; useful for debugging)
        "clean_top1_label": clean_top1_label,
        "clean_top1_idx": clean_top1_idx,
        "clean_top1_prob": clean_top1_prob,

        # hc
        "hc_top1_label": hc_top1_label,
        "hc_top1_idx": hc_top1_idx,
        "hc_top1_prob": hc_top1_prob,
        "hc_success": hc_success,
        "hc_changed_pred": hc_changed_pred,
        "hc_linf": hc_linf,
        "hc_l2": hc_l2,
        "hc_num_pixels_changed": hc_num_pix,
        "hc_changed_perc": hc_perc,
    }

def to_jsonable(obj):

    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def attack_one_image(item, model, csv_file, json_file, writer, labels_to_index, IMG_OUTDIR):
    image_path = "images/" + item["image"]
    target_label = item["label"]
    metrics = {}

    print(f"Loaded image: {image_path}")
    print(f"Target label: {target_label}")

    img = load_img(image_path)
    # plt.imshow(img)
    # plt.title("Original image")
    # plt.show()

    img_array = img_to_array(img)
    seed = img_array.copy()


    # Print baseline top-5 predictions
    print("\nBaseline predictions (top-5):")
    preds = model.predict(np.expand_dims(seed, axis=0))
    for cl in decode_predictions(preds, top=5)[0]:
        print(f"{cl[1]:20s}  prob={cl[2]:.5f}")

    t0 = time.perf_counter()
    # Run hill climbing attack
    final_img, final_fitness = hill_climb(
        initial_seed=seed,
        model=model,
        target_label=target_label,
        epsilon=0.3,
        iterations=100
    )

    hc_runtime = time.perf_counter() - t0
    metrics["runtime"] = hc_runtime
    print("\nFinal fitness:", final_fitness)

    # plt.imshow(array_to_img(final_img))
    # plt.title(f"Adversarial Result — fitness={final_fitness:.4f}")
    # plt.show()

    # -----------------------------
    # Save adversarial image
    # -----------------------------
    image_name = Path(item["image"]).stem
    hc_img_path = IMG_OUTDIR / f"{image_name}_hc.png"

    array_to_img(final_img).save(hc_img_path)

    # Print final predictions
    final_preds = model.predict(np.expand_dims(final_img, axis=0))
    print("\nFinal predictions:")
    for cl in decode_predictions(final_preds, top=5)[0]:
        print(cl)
    hc_metrics = compute_hc_eval_metrics(
        seed_img=seed,
        adv_img=final_img,
        model=model,
        human_label=target_label,
        label_to_index=labels_to_index
    )
    metrics.update(hc_metrics)
    metrics["hc_runtime_s"] = hc_runtime

    row = {
        "image_file": item["image"],
        "human_label": target_label,

        "clean_top1_label": metrics["clean_top1_label"],
        "clean_top1_idx": metrics["clean_top1_idx"],
        "clean_top1_prob": metrics["clean_top1_prob"],

        "hc_top1_label": metrics["hc_top1_label"],
        "hc_top1_idx": metrics["hc_top1_idx"],
        "hc_top1_prob": metrics["hc_top1_prob"],

        "hc_success": metrics["hc_success"],
        "hc_changed_pred": metrics["hc_changed_pred"],

        "hc_linf": metrics["hc_linf"],
        "hc_l2": metrics["hc_l2"],
        "hc_num_pixels_changed": metrics["hc_num_pixels_changed"],
        "hc_changed_perc": metrics["hc_changed_perc"],

        "hc_runtime_s": metrics["hc_runtime_s"],
    }
    writer.writerow(row)
    csv_file.flush()

    record = {
        "image": item["image"],
        "human_label": target_label,
        "hc": {
            "runtime_s": metrics["hc_runtime_s"],
            "top1": {"idx": metrics["hc_top1_idx"], "label": metrics["hc_top1_label"], "prob": metrics["hc_top1_prob"]},
            "top5": to_jsonable(decode_predictions(final_preds, top=5)[0]),
            "success": metrics["hc_success"],
            "changed_pred": metrics["hc_changed_pred"],
            "perturbation": {
                "max_change": metrics["hc_linf"],
                "l2": metrics["hc_l2"],
                "num_pixels_changed": metrics["hc_num_pixels_changed"],
                "percentage_pixels_changed": metrics["hc_changed_perc"],
            }
        }
    }
    json_file.write(json.dumps(record) + "\n")
    json_file.flush()

    return metrics



if __name__ == "__main__":
    # Load classifier
    model = vgg16.VGG16(weights="imagenet")

    # Load JSON describing dataset
    with open("data/image_labels.json") as f:
        image_list = json.load(f)

    with open("data/imagenet_classes.txt", "r") as f:
        imagenet_labels = [s.strip() for s in f.readlines()]

    label_to_index = {label: i for i, label in enumerate(imagenet_labels)}

    OUTDIR = Path("hc_results")
    os.makedirs(OUTDIR, exist_ok=True)

    CSV_PATH = OUTDIR / "attack_stats_hc.csv"
    JSONL_PATH = OUTDIR / "attack_stats_hc.jsonl"

    IMG_OUTDIR = OUTDIR / "images"
    IMG_OUTDIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_file", "human_label", "clean_top1_label", "clean_top1_idx",
        "clean_top1_prob", "hc_top1_label", "hc_top1_idx", "hc_top1_prob",
        "hc_success", "hc_changed_pred",
        "hc_linf", "hc_l2", "hc_num_pixels_changed", "hc_changed_perc",
        "hc_runtime_s",
    ]

    csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    jsonl_file = open(JSONL_PATH, "w", encoding="utf-8")

    print(f"Writing CSV to:   {CSV_PATH}")
    print(f"Writing JSONL to: {JSONL_PATH}")


    for item in tqdm(image_list, desc="Running Black-Box Attacks"):
        attack_one_image(item, model,csv_file, jsonl_file, writer, label_to_index, IMG_OUTDIR)


