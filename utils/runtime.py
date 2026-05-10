import os
import csv
from bisect import bisect_right
import yaml
import torch
from torch.utils.data import random_split, ConcatDataset

from datasets.loader import CFRDataset
from models.baseline_models import BaselineModel
from utils.seed import set_seed


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["train"]["device"] = resolve_device(config)
    return config


def resolve_device(config):
    requested_device = config["train"].get("device", "auto")
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested_device


def build_dataset(config, scene="InF_DH", sync_err=0, force_reload=False):
    base_station_ids = get_base_station_ids(config)
    return CFRDataset(
        scene=scene,
        sync_err=sync_err,
        root_dir=config["data"]["raw_path"],
        cache_dir=config["data"]["cache_path"],
        seq_len=config["data"]["sequence_length"],
        step=config["data"]["step_size"],
        base_station_ids=base_station_ids,
        force_reload=force_reload,
    )


def build_mixed_dataset(config, force_reload=False):
    """
    Build a ConcatDataset containing all scenes and sync_errors defined in config.
    """
    datasets = []
    skipped = []
    scenes = config["data"].get("scenes", ["InF_DH", "InF_DL"])
    sync_errors = config["data"].get("sync_errors", [0, 2, 10, 50])
    skip_invalid = config["data"].get("skip_invalid_datasets", True)
    
    for scene in scenes:
        for err in sync_errors:
            try:
                ds = build_dataset(config, scene=scene, sync_err=err, force_reload=force_reload)
                datasets.append(ds)
            except Exception as exc:
                if not skip_invalid:
                    raise
                skipped.append((scene, err, str(exc)))
                print(f"Skipping dataset {scene}/sync_err_{err}: {exc}")

    if not datasets:
        raise RuntimeError("No valid datasets were loaded.")

    if skipped:
        print(f"Skipped {len(skipped)} invalid dataset(s).")

    return ConcatDataset(datasets)


def build_model(config):
    return BaselineModel(
        feature_dim=get_input_feature_dim(config),
        cnn_out_channels=config["model"]["cnn"]["out_channels"],
        lstm_hidden_size=config["model"]["lstm"]["hidden_size"],
        lstm_num_layers=config["model"]["lstm"].get("num_layers", 2),
        output_dim=config["model"]["fc"]["output_dim"],
        dropout_rate=config["model"].get("dropout_rate", 0.3),
        fc_hidden_dim=config["model"]["fc"].get("hidden_dim", 128),
        use_feature_attention=config["model"].get("use_feature_attention", False),
        attention_reduction=config["model"].get("feature_attention", {}).get("reduction", 8),
    )


def get_base_station_ids(config):
    explicit_ids = config["data"].get("base_station_ids")
    if explicit_ids:
        return list(explicit_ids)

    num_base_stations = config["data"].get("num_base_stations", 1)
    return list(range(1, num_base_stations + 1))


def get_input_feature_dim(config):
    feature_dim_per_bs = config["data"].get("feature_dim_per_bs")
    if feature_dim_per_bs is None:
        feature_dim_per_bs = config["data"].get("feature_dim", 416)
    return feature_dim_per_bs * len(get_base_station_ids(config))


def create_data_splits(dataset, config):
    total_size = len(dataset)
    train_ratio = config["train"].get("train_ratio", 0.7)
    val_ratio = config["train"].get("val_ratio", 0.15)

    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(config["train"].get("seed", 42))
    return random_split(dataset, [train_size, val_size, test_size], generator=generator)


def _resolve_metadata(dataset, idx):
    if isinstance(dataset, CFRDataset):
        return dataset.get_metadata(idx)

    if isinstance(dataset, ConcatDataset):
        dataset_idx = bisect_right(dataset.cumulative_sizes, idx)
        prev_cum_size = 0 if dataset_idx == 0 else dataset.cumulative_sizes[dataset_idx - 1]
        sample_idx = idx - prev_cum_size
        return _resolve_metadata(dataset.datasets[dataset_idx], sample_idx)

    raise TypeError(f"Unsupported dataset type for metadata export: {type(dataset)}")


def export_subset_manifest(subset, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fieldnames = [
        "subset_index",
        "scene",
        "sync_err",
        "base_station_ids",
        "sequence_index",
        "start_index",
        "end_index",
        "target_index",
    ]

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for subset_index, original_index in enumerate(subset.indices):
            metadata = _resolve_metadata(subset.dataset, original_index)
            writer.writerow({"subset_index": subset_index, **metadata})


def export_split_manifests(train_dataset, val_dataset, test_dataset, config):
    output_dir = config["train"].get("output_dir", "outputs")
    split_dir = os.path.join(output_dir, "splits")
    export_subset_manifest(train_dataset, os.path.join(split_dir, "train_manifest.csv"))
    export_subset_manifest(val_dataset, os.path.join(split_dir, "val_manifest.csv"))
    export_subset_manifest(test_dataset, os.path.join(split_dir, "test_manifest.csv"))


def prepare_runtime(config_path="config/config.yaml"):
    config = load_config(config_path)
    set_seed(config["train"].get("seed", 42))
    return config


def load_checkpoint(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
