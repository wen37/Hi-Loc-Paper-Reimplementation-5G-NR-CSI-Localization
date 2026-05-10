import copy
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trainers.trainer import Trainer
from training_ablation.ablation_model import ToggleBaselineModel
from utils.runtime import (
    build_mixed_dataset,
    create_data_splits,
    load_checkpoint,
    load_config,
)
from utils.seed import set_seed
from utils.visualization import (
    plot_error_cdf,
    plot_error_histogram,
    plot_prediction_scatter,
    save_json,
)


VARIANTS = {
    "no_stabilization": {
        "label": "无稳定化",
        "use_input_norm": False,
        "use_dropout": False,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "grad_clip": None,
        "source": "existing_result1",
    },
    "norm_dropout_only": {
        "label": "仅归一化+Dropout",
        "use_input_norm": True,
        "use_dropout": True,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "grad_clip": None,
        "source": "rerun",
    },
    "optimizer_stabilization_only": {
        "label": "仅优化层稳定化",
        "use_input_norm": False,
        "use_dropout": False,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "source": "rerun",
    },
    "full_stabilization": {
        "label": "完整稳定化",
        "use_input_norm": True,
        "use_dropout": True,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "source": "existing_result2",
    },
}

ORDER = [
    "no_stabilization",
    "norm_dropout_only",
    "optimizer_stabilization_only",
    "full_stabilization",
]

EXISTING_RESULT1 = PROJECT_ROOT / "result1-直接多基站，但是效果比单基站差"
EXISTING_RESULT2 = PROJECT_ROOT / "result2-多基站，加了归一化正则化，结果好了很多。没加注意力机制"


def clone_base_config():
    config = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    config["model"]["use_feature_attention"] = False
    return config


def build_model(config, variant):
    feature_dim = config["data"]["feature_dim_per_bs"] * len(config["data"]["base_station_ids"])
    return ToggleBaselineModel(
        feature_dim=feature_dim,
        cnn_out_channels=config["model"]["cnn"]["out_channels"],
        lstm_hidden_size=config["model"]["lstm"]["hidden_size"],
        lstm_num_layers=config["model"]["lstm"].get("num_layers", 2),
        output_dim=config["model"]["fc"]["output_dim"],
        dropout_rate=config["model"].get("dropout_rate", 0.3),
        fc_hidden_dim=config["model"]["fc"].get("hidden_dim", 128),
        use_input_norm=variant["use_input_norm"],
        use_dropout=variant["use_dropout"],
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


def prepare_variant_config(base_config, variant_name, variant):
    config = copy.deepcopy(base_config)
    config["train"]["output_dir"] = str(CURRENT_DIR / "results" / variant_name)
    config["train"]["checkpoint_path"] = str(
        CURRENT_DIR / "results" / variant_name / "checkpoints" / "best_model.pt"
    )
    config["train"]["lr"] = variant["lr"]
    config["train"]["weight_decay"] = variant["weight_decay"]
    config["train"]["grad_clip"] = variant["grad_clip"]
    return config


def load_existing_metrics(result_dir, variant_name, variant):
    metrics_path = result_dir / "outputs" / "test" / "metrics.json"
    train_curve = result_dir / "outputs" / "train" / "loss_curve.png"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    metrics.update(
        {
            "variant": variant_name,
            "label": variant["label"],
            "lr": variant["lr"],
            "weight_decay": variant["weight_decay"],
            "grad_clip": variant["grad_clip"],
            "use_input_norm": variant["use_input_norm"],
            "use_dropout": variant["use_dropout"],
            "source": variant["source"],
        }
    )
    variant_dir = CURRENT_DIR / "results" / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, str(variant_dir / "test" / "metrics.json"))
    if train_curve.exists():
        (variant_dir / "train").mkdir(parents=True, exist_ok=True)
        shutil.copy2(train_curve, variant_dir / "train" / "loss_curve.png")
    return metrics


def run_variant(variant_name, force_reload=False):
    variant = VARIANTS[variant_name]
    config = prepare_variant_config(clone_base_config(), variant_name, variant)
    set_seed(config["train"].get("seed", 42))

    print(f"\n=== Running {variant_name} ({variant['label']}) ===")
    dataset = build_mixed_dataset(config, force_reload=force_reload)
    train_dataset, val_dataset, test_dataset = create_data_splits(dataset, config)
    print(
        f"Dataset split -> train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}"
    )

    train_loader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["train"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["train"]["batch_size"], shuffle=False)

    model = build_model(config, variant)
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.fit()

    checkpoint = load_checkpoint(model, config["train"]["checkpoint_path"], config["train"]["device"])
    mse, rmse, mae, mean_distance_error, errors, y_true, y_pred = evaluate(
        model, test_loader, config["train"]["device"]
    )
    metrics = {
        "variant": variant_name,
        "label": variant["label"],
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_val_loss": float(checkpoint["val_loss"]),
        "test_mse": float(mse),
        "test_rmse": float(rmse),
        "test_mae": float(mae),
        "mean_distance_error": float(mean_distance_error),
        "p50_distance_error": float(np.percentile(errors, 50)),
        "p90_distance_error": float(np.percentile(errors, 90)),
        "p95_distance_error": float(np.percentile(errors, 95)),
        "lr": variant["lr"],
        "weight_decay": variant["weight_decay"],
        "grad_clip": variant["grad_clip"],
        "use_input_norm": variant["use_input_norm"],
        "use_dropout": variant["use_dropout"],
        "source": variant["source"],
    }
    save_test_outputs(config["train"]["output_dir"], metrics, errors, y_true, y_pred)
    return metrics


def save_summary(metrics_list):
    results_dir = CURRENT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(metrics_list, key=lambda x: ORDER.index(x["variant"]))
    csv_path = results_dir / "training_ablation_summary.csv"
    md_path = results_dir / "training_ablation_summary.md"
    fieldnames = [
        "variant",
        "label",
        "use_input_norm",
        "use_dropout",
        "lr",
        "weight_decay",
        "grad_clip",
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
        f.write("| 方案 | 归一化 | Dropout | lr | wd | grad clip | Epoch | Val Loss | RMSE | MAE | Mean Dist | P90 | 来源 |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            grad_clip = "-" if row["grad_clip"] is None else row["grad_clip"]
            f.write(
                f"| {row['label']} | {row['use_input_norm']} | {row['use_dropout']} | "
                f"{row['lr']:.4f} | {row['weight_decay']:.4f} | {grad_clip} | "
                f"{row['checkpoint_epoch']} | {row['checkpoint_val_loss']:.4f} | "
                f"{row['test_rmse']:.4f} | {row['test_mae']:.4f} | "
                f"{row['mean_distance_error']:.4f} | {row['p90_distance_error']:.4f} | {row['source']} |\n"
            )

    plot_metric(rows, results_dir / "training_ablation_rmse.png", "test_rmse", "RMSE")
    plot_metric(rows, results_dir / "training_ablation_mde.png", "mean_distance_error", "Mean Distance Error")


def plot_metric(rows, save_path, metric_key, ylabel):
    labels = [row["label"] for row in rows]
    values = [row[metric_key] for row in rows]
    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values, alpha=0.85)
    plt.ylabel(ylabel)
    plt.title(f"Training Ablation - {ylabel}")
    plt.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    force_reload = "--force-reload" in sys.argv
    metrics_list = []

    metrics_list.append(load_existing_metrics(EXISTING_RESULT1, "no_stabilization", VARIANTS["no_stabilization"]))
    metrics_list.append(run_variant("norm_dropout_only", force_reload=force_reload))
    metrics_list.append(run_variant("optimizer_stabilization_only", force_reload=force_reload))
    metrics_list.append(load_existing_metrics(EXISTING_RESULT2, "full_stabilization", VARIANTS["full_stabilization"]))

    save_summary(metrics_list)
    print("\nTraining ablation finished.")
    print(f"Summary saved to: {CURRENT_DIR / 'results' / 'training_ablation_summary.csv'}")


if __name__ == "__main__":
    main()
