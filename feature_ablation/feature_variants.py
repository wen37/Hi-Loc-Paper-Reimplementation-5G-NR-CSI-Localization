import numpy as np
from scipy.stats import kurtosis, skew


FEATURE_VARIANTS = {
    "downsample_only": {
        "label": "仅降采样特征",
        "feature_dim_per_bs": 408,
    },
    "stats_only": {
        "label": "仅统计特征",
        "feature_dim_per_bs": 8,
    },
    "downsample_plus_stats": {
        "label": "降采样+统计特征",
        "feature_dim_per_bs": 416,
    },
}


def _as_complex_array(cfr):
    array = np.asarray(cfr)

    if np.iscomplexobj(array):
        return array

    if array.dtype.names:
        field_names = list(array.dtype.names)
        if {"real", "imag"}.issubset(field_names):
            return array["real"] + 1j * array["imag"]
        if len(field_names) >= 2:
            return array[field_names[0]] + 1j * array[field_names[1]]

    if array.ndim > 0 and array.shape[-1] == 2 and np.issubdtype(array.dtype, np.number):
        return array[..., 0] + 1j * array[..., 1]

    return array.astype(np.complex64)


def _collapse_to_subcarrier_vector(cfr):
    cfr = _as_complex_array(cfr)
    if cfr.ndim == 1:
        return cfr

    reduce_axes = tuple(range(cfr.ndim - 1))
    return np.mean(cfr, axis=reduce_axes)


def _compute_stats(magnitude):
    f_mean = float(np.mean(magnitude))
    f_std = float(np.std(magnitude))
    f_max = float(np.max(magnitude))
    f_min = float(np.min(magnitude))
    f_skew = float(np.nan_to_num(skew(magnitude, axis=None, bias=False), nan=0.0))
    f_kurt = float(np.nan_to_num(kurtosis(magnitude, axis=None, bias=False), nan=0.0))
    f_energy = float(np.sum(magnitude ** 2))
    prob = magnitude / (np.sum(magnitude) + 1e-12)
    f_entropy = float(-np.sum(prob * np.log(prob + 1e-12)))
    return np.array(
        [f_mean, f_std, f_max, f_min, f_skew, f_kurt, f_energy, f_entropy],
        dtype=np.float32,
    )


def extract_feature_variant(cfr_complex, mode="downsample_plus_stats", downsample_rate=8):
    if mode not in FEATURE_VARIANTS:
        raise ValueError(f"Unsupported feature mode: {mode}")

    cfr_vector = _collapse_to_subcarrier_vector(cfr_complex)
    downsampled = np.abs(cfr_vector[::downsample_rate]).astype(np.float32)
    magnitude = np.abs(cfr_vector)
    stats = _compute_stats(magnitude)

    if mode == "downsample_only":
        return downsampled
    if mode == "stats_only":
        return stats
    return np.concatenate([downsampled, stats], axis=0)
