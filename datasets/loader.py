import os
import numpy as np
import h5py
import scipy.io as sio
from scipy.io.matlab import MatReadError
import torch
from torch.utils.data import Dataset
from .feature_engineering import extract_enhanced_features
from .sequence_builder import build_sequences

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

    # MATLAB v7.3 复数数组经常保存为 compound dtype: [('real', ...), ('imag', ...)]
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

    # UE_pos 这类小维度标签常见存储为 (3, N)，转成 (N, 3)
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
                f"Unable to find dataset '{key}' in {file_path}. "
                f"Available datasets: {available}"
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
                f"Unable to find variable '{key}' in {file_path}. "
                f"Available variables: {available}"
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
                f"Failed to read MAT file: {file_path}. "
                "It is neither a valid HDF5 MAT(v7.3) file nor a readable legacy MAT file. "
                "The file may be truncated, corrupted, or incompletely copied."
            ) from legacy_exc

class CFRDataset(Dataset):
    def __init__(self, scene='InF_DH', sync_err=0, root_dir='simulated_dataset', 
                 cache_dir='data/cache', seq_len=5, step=2, base_station_ids=None,
                 force_reload=False):
        self.scene = scene
        self.sync_err = sync_err
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.seq_len = seq_len
        self.step = step
        self.base_station_ids = list(base_station_ids or [1])

        if len(self.base_station_ids) == 1:
            cache_bs_tag = f"bs{self.base_station_ids[0]}"
        else:
            cache_bs_tag = f"bs{self.base_station_ids[0]}to{self.base_station_ids[-1]}"
        
        self.cache_path_x = os.path.join(
            cache_dir, f"{scene}_sync{sync_err}_{cache_bs_tag}_X.npy"
        )
        self.cache_path_y = os.path.join(
            cache_dir, f"{scene}_sync{sync_err}_{cache_bs_tag}_Y.npy"
        )
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        if force_reload or not os.path.exists(self.cache_path_x):
            self._process_raw_data()
        else:
            self.data_x = np.load(self.cache_path_x)
            self.data_y = np.load(self.cache_path_y)

        self.metadata = self._build_metadata(len(self.data_x))

    def _build_metadata(self, num_sequences):
        metadata = []
        for sequence_index in range(num_sequences):
            start_index = sequence_index * self.step
            end_index = start_index + self.seq_len - 1
            metadata.append(
                {
                    "scene": self.scene,
                    "sync_err": self.sync_err,
                    "base_station_ids": "-".join(map(str, self.base_station_ids)),
                    "sequence_index": sequence_index,
                    "start_index": start_index,
                    "end_index": end_index,
                    "target_index": end_index,
                }
            )
        return metadata
            
    def _process_raw_data(self):
        print(f"Processing raw data for {self.scene} sync_err_{self.sync_err}...")
        scene_path = os.path.join(self.root_dir, self.scene, f"sync_err_{self.sync_err}")
        
        # 1. Load UE positions (2000, 3)
        ue_pos_path = os.path.join(scene_path, "UE_pos.mat")
        ue_pos = load_mat_auto(ue_pos_path, 'UE_pos')
        if ue_pos is None:
            raise ValueError(f"Could not load UE_pos from {ue_pos_path}")

        all_bs_features = []
        for base_station_id in self.base_station_ids:
            cfr_path = os.path.join(scene_path, f"CFR{base_station_id}.mat")
            if not os.path.exists(cfr_path):
                raise FileNotFoundError(f"Base station file not found: {cfr_path}")

            cfr_data = load_mat_auto(cfr_path, 'CFR', expected_rows=ue_pos.shape[0])
            if cfr_data is None:
                raise ValueError(f"Could not load CFR data from {cfr_path}")

            print(
                f"Loaded BS{base_station_id}: cfr shape={cfr_data.shape}, "
                f"ue_pos shape={ue_pos.shape}"
            )

            bs_features = []
            for i in range(cfr_data.shape[0]):
                feat = extract_enhanced_features(cfr_data[i])
                bs_features.append(feat)

            all_bs_features.append(np.array(bs_features, dtype=np.float32))

        X_raw = np.concatenate(all_bs_features, axis=1)
        Y_raw = ue_pos[:, :2] # (2000, 2) - Taking (x, y)
        
        # 3. Build sequences
        self.data_x, self.data_y = build_sequences(X_raw, Y_raw, self.seq_len, self.step)
        
        # 4. Save to cache
        np.save(self.cache_path_x, self.data_x)
        np.save(self.cache_path_y, self.data_y)
        print(f"Saved to cache: {self.cache_path_x}")

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data_x[idx]).float()
        y = torch.from_numpy(self.data_y[idx]).float()
        return x, y

    def get_metadata(self, idx):
        return self.metadata[idx]
