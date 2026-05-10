import json
import os

import matplotlib.pyplot as plt
import numpy as np


def ensure_parent_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def save_json(data, file_path):
    ensure_parent_dir(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def plot_loss_curves(train_losses, val_losses, save_path):
    ensure_parent_dir(save_path)
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_error_histogram(errors, save_path, bins=30):
    ensure_parent_dir(save_path)
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=bins, edgecolor="black", alpha=0.8)
    plt.xlabel("Distance Error")
    plt.ylabel("Count")
    plt.title("Test Error Histogram")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_error_cdf(errors, save_path):
    ensure_parent_dir(save_path)
    sorted_errors = np.sort(np.asarray(errors))
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

    plt.figure(figsize=(8, 5))
    plt.plot(sorted_errors, cdf)
    plt.xlabel("Distance Error")
    plt.ylabel("CDF")
    plt.title("Test Error CDF")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_prediction_scatter(y_true, y_pred, save_path):
    ensure_parent_dir(save_path)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true[:, 0], y_true[:, 1], s=12, label="Ground Truth", alpha=0.7)
    plt.scatter(y_pred[:, 0], y_pred[:, 1], s=12, label="Prediction", alpha=0.7)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Ground Truth vs Prediction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
