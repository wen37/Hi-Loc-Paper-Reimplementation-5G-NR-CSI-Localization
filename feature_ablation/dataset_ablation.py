import os

import h5py
import numpy as np
import scipy.io as sio
import torch
from scipy.io.matlab import MatReadError
from torch.utils.data import ConcatDataset, Dataset

from datasets.sequence_builder import build_sequences
from feature_ablation.feature_variants import extract_feature_variant


def _collect_datasets(group, prefix=""):
    datasets = {}
    for name, obj in group.items():
        if name.startswith("#"):
            continue
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(obj, h5py.Dataset):
            datasets[path] = obj
        elif isinstance(obj, h5py.Group):
            datasets.update(_collect_datasets(obj, path))
    return datasets


def _to_numpy(dataset):
    array = np.array(dataset)
    if array.dtype.names and {"real", "imag"}.issubset(array.dtype.names):
        array = array["real"] + 1j * array["imag"]
    return array


def _maybe_transpose(array, expected_rows=None):
    if array.ndim != 2:
        return array
    if expected_rows is not None:
        if array.shape[0] != expected_rows and array.shape[1] == expected_rows:
            return array.T
        return array
    if array.shape[0] <= 16 and array.shape[1] > array.shape[0]:
        return array.T
    return array


def _select_h5_dataset(datasets, file_path, key=None):
    selected = None
    if key is not None:
        for path, dataset in datasets.items():
            if path == key or path.split("/")[-1] == key:
                selected = dataset
                break
        if selected is None:
            key_lower = key.lower()
            for path, dataset in datasets.items():
                basename = path.split("/")[-1].lower()
                if basename == key_lower or key_lower in basename:
                    selected = dataset
                    break

    if selected is None:
        if len(datasets) == 1:
            selected = next(iter(datasets.values()))
        else:
            available = ", ".join(datasets.keys())
            raise KeyError(
                f"Unable to find dataset '{key}' in {file_path}. Available datasets: {available}"
            )
    return selected


def _load_h5_mat(file_path, key=None, expected_rows=None):
    with h5py.File(file_path, "r") as f:
        datasets = _collect_datasets(f)
        if not datasets:
            raise ValueError(f"No datasets found in {file_path}")
        selected = _select_h5_dataset(datasets, file_path, key=key)
        array = _to_numpy(selected)
        return _maybe_transpose(array, expected_rows=expected_rows)


def _load_legacy_mat(file_path, key=None, expected_rows=None):
    data = sio.loadmat(file_path)
    candidates = {k: v for k, v in data.items() if not k.startswith("__")}
    if not candidates:
        raise ValueError(f"No variables found in {file_path}")

    selected = None
    if key is not None:
        for name, value in candidates.items():
            if name == key:
                selected = value
                break
        if selected is None:
            key_lower = key.lower()
            for name, value in candidates.items():
                if name.lower() == key_lower or key_lower in name.lower():
                    selected = value
                    break

    if selected is None:
        if len(candidates) == 1:
            selected = next(iter(candidates.values()))
        else:
            available = ", ".join(candidates.keys())
            raise KeyError(
                f"Unable to find variable '{key}' in {file_path}. Available variables: {available}"
            )
    return _maybe_transpose(np.asarray(selected), expected_rows=expected_rows)


def load_mat_auto(file_path, key=None, expected_rows=None):
    try:
        return _load_h5_mat(file_path, key=key, expected_rows=expected_rows)
    except OSError as exc:
        if "file signature not found" not in str(exc).lower():
            raise
        try:
            return _load_legacy_mat(file_path, key=key, expected_rows=expected_rows)
        except MatReadError as legacy_exc:
            raise ValueError(
                f"Failed to read MAT file: {file_path}. The file may be truncated, corrupted, or incompletely copied."
            ) from legacy_exc


class FeatureAblationDataset(Dataset):
    def __init__(
        self,
        scene,
        sync_err,
        root_dir,
        cache_dir,
        seq_len,
        step,
        base_station_ids,
        feature_mode,
        downsample_rate=8,
        force_reload=False,
    ):
        self.scene = scene
        self.sync_err = sync_err
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.seq_len = seq_len
        self.step = step
        self.base_station_ids = list(base_station_ids)
        self.feature_mode = feature_mode
        self.downsample_rate = downsample_rate

        bs_tag = (
            f"bs{self.base_station_ids[0]}"
            if len(self.base_station_ids) == 1
            else f"bs{self.base_station_ids[0]}to{self.base_station_ids[-1]}"
        )
        mode_tag = f"{feature_mode}_ds{downsample_rate}"
        self.cache_path_x = os.path.join(
            cache_dir, f"{scene}_sync{sync_err}_{bs_tag}_{mode_tag}_X.npy"
        )
        self.cache_path_y = os.path.join(
            cache_dir, f"{scene}_sync{sync_err}_{bs_tag}_{mode_tag}_Y.npy"
        )

        os.makedirs(cache_dir, exist_ok=True)
        if force_reload or not os.path.exists(self.cache_path_x):
            self._process_raw_data()
        else:
            self.data_x = np.load(self.cache_path_x)
            self.data_y = np.load(self.cache_path_y)

    def _process_raw_data(self):
        print(
            f"[{self.feature_mode}] Processing raw data for {self.scene} sync_err_{self.sync_err}..."
        )
        scene_path = os.path.join(self.root_dir, self.scene, f"sync_err_{self.sync_err}")
        ue_pos = load_mat_auto(os.path.join(scene_path, "UE_pos.mat"), "UE_pos")

        all_bs_features = []
        for base_station_id in self.base_station_ids:
            cfr_path = os.path.join(scene_path, f"CFR{base_station_id}.mat")
            cfr_data = load_mat_auto(cfr_path, "CFR", expected_rows=ue_pos.shape[0])
            bs_features = [
                extract_feature_variant(
                    cfr_data[i],
                    mode=self.feature_mode,
                    downsample_rate=self.downsample_rate,
                )
                for i in range(cfr_data.shape[0])
            ]
            all_bs_features.append(np.asarray(bs_features, dtype=np.float32))

        x_raw = np.concatenate(all_bs_features, axis=1)
        y_raw = ue_pos[:, :2]
        self.data_x, self.data_y = build_sequences(x_raw, y_raw, self.seq_len, self.step)
        np.save(self.cache_path_x, self.data_x)
        np.save(self.cache_path_y, self.data_y)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.data_x[idx]).float(),
            torch.from_numpy(self.data_y[idx]).float(),
        )


def build_mixed_ablation_dataset(config, feature_mode, force_reload=False):
    datasets = []
    skipped = []
    scenes = config["data"].get("scenes", ["InF_DH", "InF_DL"])
    sync_errors = config["data"].get("sync_errors", [0, 10, 50])
    skip_invalid = config["data"].get("skip_invalid_datasets", True)
    for scene in scenes:
        for err in sync_errors:
            try:
                datasets.append(
                    FeatureAblationDataset(
                        scene=scene,
                        sync_err=err,
                        root_dir=config["data"]["raw_path"],
                        cache_dir=config["data"]["cache_path"],
                        seq_len=config["data"]["sequence_length"],
                        step=config["data"]["step_size"],
                        base_station_ids=config["data"]["base_station_ids"],
                        feature_mode=feature_mode,
                        downsample_rate=config["data"].get("downsample_rate", 8),
                        force_reload=force_reload,
                    )
                )
            except Exception as exc:
                if not skip_invalid:
                    raise
                skipped.append((scene, err, str(exc)))
                print(f"Skipping dataset {scene}/sync_err_{err}: {exc}")

    if not datasets:
        raise RuntimeError("No valid datasets were loaded for feature ablation.")
    if skipped:
        print(f"Skipped {len(skipped)} invalid dataset(s).")
    return ConcatDataset(datasets)
