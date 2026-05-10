import copy
import csv
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_ablation.dataset_ablation import build_mixed_ablation_dataset
from feature_ablation.feature_variants import FEATURE_VARIANTS
from models.baseline_models import BaselineModel
from trainers.trainer import Trainer
from utils.runtime import load_checkpoint, load_config
from utils.seed import set_seed
from utils.visualization import (
    plot_error_cdf,
    plot_error_histogram,
    plot_prediction_scatter,
    save_json,
)


VARIANT_ORDER = [
    "downsample_only",
    "stats_only",
    "downsample_plus_stats",
]

# Reuse the already finished stable baseline result to avoid redundant retraining.
EXISTING_BASELINE_METRICS = PROJECT_ROOT / "result2-多基站，加了归一化正则化，结果好了很多。没加注意力机制" / "outputs" / "test" / "metrics.json"
EXISTING_BASELINE_TRAIN = PROJECT_ROOT / "result2-多基站，加了归一化正则化，结果好了很多。没加注意力机制" / "outputs" / "train" / "loss_curve.png"


def create_data_splits(dataset, config):
    total_size = len(dataset)
    train_ratio = config["train"].get("train_ratio", 0.7)
    val_ratio = config["train"].get("val_ratio", 0.15)

    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(config["train"].get("seed", 42))
    return random_split(dataset, [train_size, val_size, test_size], generator=generator)


def build_model(config, feature_dim):
    return BaselineModel(
        feature_dim=feature_dim,
        cnn_out_channels=config["model"]["cnn"]["out_channels"],
        lstm_hidden_size=config["model"]["lstm"]["hidden_size"],
        lstm_num_layers=config["model"]["lstm"].get("num_layers", 2),
        output_dim=config["model"]["fc"]["output_dim"],
        dropout_rate=config["model"].get("dropout_rate", 0.3),
        fc_hidden_dim=config["model"]["fc"].get("hidden_dim", 128),
        use_feature_attention=False,
        attention_reduction=config["model"].get("feature_attention", {}).get("reduction", 8),
    )


def evaluate(model, data_loader, device):
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    dist_sum = 0.0
    sample_count = 0
    all_errors = []
    all_true = []
    all_pred = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            batch_size = x.size(0)

            mse_sum += torch.sum((pred - y) ** 2).item()
            mae_sum += torch.sum(torch.abs(pred - y)).item()
            batch_errors = torch.norm(pred - y, dim=1)
            dist_sum += torch.sum(batch_errors).item()
            sample_count += batch_size
            all_errors.extend(batch_errors.cpu().tolist())
            all_true.extend(y.cpu().tolist())
            all_pred.extend(pred.cpu().tolist())

    mse = mse_sum / sample_count
    rmse = mse ** 0.5
    mae = mae_sum / (sample_count * 2)
    mean_distance_error = dist_sum / sample_count
    return mse, rmse, mae, mean_distance_error, all_errors, all_true, all_pred


def save_test_outputs(output_dir, metrics, errors, y_true, y_pred):
    test_dir = Path(output_dir) / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    save_json(metrics, str(test_dir / "metrics.json"))
    plot_error_histogram(errors, str(test_dir / "error_histogram.png"))
    plot_error_cdf(errors, str(test_dir / "error_cdf.png"))
    plot_prediction_scatter(y_true, y_pred, str(test_dir / "prediction_scatter.png"))

    with open(test_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "true_x", "true_y", "pred_x", "pred_y", "distance_error"])
        for idx, (true_xy, pred_xy, error) in enumerate(zip(y_true, y_pred, errors)):
            writer.writerow([idx, true_xy[0], true_xy[1], pred_xy[0], pred_xy[1], error])


def clone_base_config():
    config = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    config["model"]["use_feature_attention"] = False
    return config


def prepare_variant_config(base_config, variant_name, feature_dim_per_bs):
    config = copy.deepcopy(base_config)
    config["data"]["cache_path"] = str(CURRENT_DIR / "cache" / variant_name)
    config["data"]["feature_dim_per_bs"] = feature_dim_per_bs
    config["train"]["output_dir"] = str(CURRENT_DIR / "results" / variant_name)
    config["train"]["checkpoint_path"] = str(
        CURRENT_DIR / "results" / variant_name / "checkpoints" / "best_model.pt"
    )
    return config


def run_variant(variant_name, force_reload=False):
    variant = FEATURE_VARIANTS[variant_name]
    config = prepare_variant_config(clone_base_config(), variant_name, variant["feature_dim_per_bs"])
    set_seed(config["train"].get("seed", 42))

    print(f"\n=== Running {variant_name} ({variant['label']}) ===")
    print("Building mixed dataset...")
    dataset = build_mixed_ablation_dataset(config, variant_name, force_reload=force_reload)
    train_dataset, val_dataset, test_dataset = create_data_splits(dataset, config)
    print(
        f"Dataset split -> train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}"
    )

    train_loader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["train"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["train"]["batch_size"], shuffle=False)

    feature_dim = variant["feature_dim_per_bs"] * len(config["data"]["base_station_ids"])
    model = build_model(config, feature_dim)
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.fit()

    checkpoint = load_checkpoint(model, config["train"]["checkpoint_path"], config["train"]["device"])
    mse, rmse, mae, mean_distance_error, errors, y_true, y_pred = evaluate(
        model, test_loader, config["train"]["device"]
    )

    metrics = {
        "variant": variant_name,
        "label": variant["label"],
        "feature_dim_per_bs": variant["feature_dim_per_bs"],
        "input_feature_dim": feature_dim,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_val_loss": float(checkpoint["val_loss"]),
        "test_mse": float(mse),
        "test_rmse": float(rmse),
        "test_mae": float(mae),
        "mean_distance_error": float(mean_distance_error),
        "p50_distance_error": float(np.percentile(errors, 50)),
        "p90_distance_error": float(np.percentile(errors, 90)),
        "p95_distance_error": float(np.percentile(errors, 95)),
        "source": "rerun",
    }
    save_test_outputs(config["train"]["output_dir"], metrics, errors, y_true, y_pred)
    return metrics


def load_existing_baseline_metrics():
    import json

    with open(EXISTING_BASELINE_METRICS, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    metrics.update(
        {
            "variant": "downsample_plus_stats",
            "label": FEATURE_VARIANTS["downsample_plus_stats"]["label"],
            "feature_dim_per_bs": FEATURE_VARIANTS["downsample_plus_stats"]["feature_dim_per_bs"],
            "input_feature_dim": FEATURE_VARIANTS["downsample_plus_stats"]["feature_dim_per_bs"] * 18,
            "source": "existing_result2",
        }
    )
    baseline_dir = CURRENT_DIR / "results" / "downsample_plus_stats"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, str(baseline_dir / "test" / "metrics.json"))
    if EXISTING_BASELINE_TRAIN.exists():
        (baseline_dir / "train").mkdir(parents=True, exist_ok=True)
        shutil.copy2(EXISTING_BASELINE_TRAIN, baseline_dir / "train" / "loss_curve.png")
    return metrics


def save_summary(metrics_list):
    summary_dir = CURRENT_DIR / "results"
    summary_dir.mkdir(parents=True, exist_ok=True)

    csv_path = summary_dir / "feature_ablation_summary.csv"
    md_path = summary_dir / "feature_ablation_summary.md"
    rows = sorted(metrics_list, key=lambda x: VARIANT_ORDER.index(x["variant"]))
    fieldnames = [
        "variant",
        "label",
        "feature_dim_per_bs",
        "input_feature_dim",
        "checkpoint_epoch",
        "checkpoint_val_loss",
        "test_mse",
        "test_rmse",
        "test_mae",
        "mean_distance_error",
        "p50_distance_error",
        "p90_distance_error",
        "p95_distance_error",
        "source",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| 方案 | 每基站特征维 | 总输入维 | Epoch | Val Loss | RMSE | MAE | Mean Dist | P50 | P90 | P95 | 来源 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row['label']} | {row['feature_dim_per_bs']} | {row['input_feature_dim']} | "
                f"{row['checkpoint_epoch']} | {row['checkpoint_val_loss']:.4f} | "
                f"{row['test_rmse']:.4f} | {row['test_mae']:.4f} | {row['mean_distance_error']:.4f} | "
                f"{row['p50_distance_error']:.4f} | {row['p90_distance_error']:.4f} | "
                f"{row['p95_distance_error']:.4f} | {row['source']} |\n"
            )

    plot_metric_bars(rows, summary_dir / "feature_ablation_rmse.png", "test_rmse", "RMSE")
    plot_metric_bars(rows, summary_dir / "feature_ablation_mde.png", "mean_distance_error", "Mean Distance Error")


def plot_metric_bars(rows, save_path, metric_key, ylabel):
    labels = [row["label"] for row in rows]
    values = [row[metric_key] for row in rows]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, alpha=0.85)
    plt.ylabel(ylabel)
    plt.title(f"Feature Ablation - {ylabel}")
    plt.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    force_reload = "--force-reload" in sys.argv
    metrics_list = []

    for variant_name in ["downsample_only", "stats_only"]:
        metrics_list.append(run_variant(variant_name, force_reload=force_reload))

    metrics_list.append(load_existing_baseline_metrics())
    save_summary(metrics_list)
    print("\nFeature ablation finished.")
    print(f"Summary saved to: {CURRENT_DIR / 'results' / 'feature_ablation_summary.csv'}")


if __name__ == "__main__":
    main()
