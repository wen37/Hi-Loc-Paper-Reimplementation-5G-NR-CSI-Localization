import numpy as np
from scipy.stats import skew, kurtosis


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

    # 当前真实数据单样本形状为 (2, 16, 3264)，这里对非子载波维做均值聚合，
    # 最终得到一条长度为 3264 的复数 CFR 向量，便于后续统一特征工程。
    reduce_axes = tuple(range(cfr.ndim - 1))
    return np.mean(cfr, axis=reduce_axes)


def extract_enhanced_features(cfr_complex, downsample_rate=8):
    """
    Extract enhanced features from complex CFR data.
    Input:
        cfr_complex: (N_subcarriers,) complex vector
    Output:
        enhanced_features: (416,) vector (408 downsampled + 8 statistical)
    """
    cfr_vector = _collapse_to_subcarrier_vector(cfr_complex)

    # 1. Frequency domain downsampling
    # 3264 / 8 = 408
    downsampled = cfr_vector[::downsample_rate]
    
    # 2. Magnitude calculation
    magnitude = np.abs(cfr_vector)
    
    # 3. Statistical features (8 features)
    f_mean = float(np.mean(magnitude))
    f_std = float(np.std(magnitude))
    f_max = float(np.max(magnitude))
    f_min = float(np.min(magnitude))
    f_skew = float(np.nan_to_num(skew(magnitude, axis=None, bias=False), nan=0.0))
    f_kurt = float(np.nan_to_num(kurtosis(magnitude, axis=None, bias=False), nan=0.0))
    f_energy = float(np.sum(magnitude ** 2))

    prob = magnitude / (np.sum(magnitude) + 1e-12)
    f_entropy = float(-np.sum(prob * np.log(prob + 1e-12)))
    
    stats = np.array(
        [f_mean, f_std, f_max, f_min, f_skew, f_kurt, f_energy, f_entropy],
        dtype=np.float32,
    )
    
    # Concatenate real/imag or magnitude for downsampled?
    # Usually magnitude is more stable for localization, but user said "复数CFR向量"
    # If we use complex, we'd have 408*2. Let's use magnitude for the 408 part to keep it 408+8=416.
    
    enhanced_features = np.concatenate(
        [np.abs(downsampled).astype(np.float32), stats],
        axis=0,
    )
    
    return enhanced_features
