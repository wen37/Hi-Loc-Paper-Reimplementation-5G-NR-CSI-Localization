import torch
import csv
import os
import numpy as np
from torch.utils.data import DataLoader

from utils.runtime import (
    prepare_runtime,
    build_mixed_dataset,
    build_model,
    create_data_splits,
    load_checkpoint,
    export_split_manifests,
)
from utils.visualization import (
    save_json,
    plot_error_histogram,
    plot_error_cdf,
    plot_prediction_scatter,
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


def save_test_outputs(config, metrics, errors, y_true, y_pred):
    output_dir = os.path.join(config["train"].get("output_dir", "outputs"), "test")
    os.makedirs(output_dir, exist_ok=True)

    save_json(metrics, os.path.join(output_dir, "metrics.json"))
    plot_error_histogram(errors, os.path.join(output_dir, "error_histogram.png"))
    plot_error_cdf(errors, os.path.join(output_dir, "error_cdf.png"))
    plot_prediction_scatter(y_true, y_pred, os.path.join(output_dir, "prediction_scatter.png"))

    csv_path = os.path.join(output_dir, "predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "true_x", "true_y", "pred_x", "pred_y", "distance_error"])
        for idx, (true_xy, pred_xy, error) in enumerate(zip(y_true, y_pred, errors)):
            writer.writerow([idx, true_xy[0], true_xy[1], pred_xy[0], pred_xy[1], error])


def main():
    config = prepare_runtime()
    device = config["train"]["device"]
    print(f"Using device: {device}")

    print("Building mixed dataset...")
    dataset = build_mixed_dataset(config)
    train_dataset, val_dataset, test_dataset = create_data_splits(dataset, config)
    export_split_manifests(train_dataset, val_dataset, test_dataset, config)
    print(f"Test dataset size: {len(test_dataset)}")
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
    )

    model = build_model(config).to(device)
    checkpoint = load_checkpoint(
        model, config["train"]["checkpoint_path"], device
    )

    mse, rmse, mae, mean_distance_error, errors, y_true, y_pred = evaluate(model, test_loader, device)
    metrics = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_val_loss": float(checkpoint["val_loss"]),
        "test_mse": float(mse),
        "test_rmse": float(rmse),
        "test_mae": float(mae),
        "mean_distance_error": float(mean_distance_error),
        "p50_distance_error": float(np.percentile(errors, 50)),
        "p90_distance_error": float(np.percentile(errors, 90)),
        "p95_distance_error": float(np.percentile(errors, 95)),
    }
    save_test_outputs(config, metrics, errors, y_true, y_pred)

    print(f"Loaded checkpoint epoch: {checkpoint['epoch']}")
    print(f"Checkpoint val loss: {checkpoint['val_loss']:.4f}")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Mean Distance Error: {mean_distance_error:.4f}")
    print(f"P50 Distance Error: {metrics['p50_distance_error']:.4f}")
    print(f"P90 Distance Error: {metrics['p90_distance_error']:.4f}")
    print(f"P95 Distance Error: {metrics['p95_distance_error']:.4f}")


if __name__ == "__main__":
    main()
