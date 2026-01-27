import hashlib
from operator import itemgetter
from typing import List, Tuple
import librosa
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (binary_erosion,
                                      generate_binary_structure,
                                      iterate_structure)

from dejavu.config.settings import (CONNECTIVITY_MASK, DEFAULT_AMP_MIN,
                                    DEFAULT_FAN_VALUE, DEFAULT_FS,
                                    DEFAULT_OVERLAP_RATIO, DEFAULT_WINDOW_SIZE,
                                    FINGERPRINT_REDUCTION, MAX_HASH_TIME_DELTA,
                                    MIN_HASH_TIME_DELTA,
                                    PEAK_NEIGHBORHOOD_SIZE, PEAK_SORT)


def fingerprint(channel_samples: np.ndarray,
                Fs: int = DEFAULT_FS,
                fan_value: int = DEFAULT_FAN_VALUE,
                amp_min: int = DEFAULT_AMP_MIN) -> List[Tuple[str, int]]:
    """
    Uses CQT for pitch invariance and Triplets for time invariance.
    """
    # 1. Constant-Q Transform (Replaces FFT for pitch invariance)

    C = np.abs(librosa.cqt(channel_samples.astype(float), 
                         sr=Fs, 
                         hop_length=512, 
                         fmin=librosa.note_to_hz('C1'), 
                         n_bins=84,      # Reduced from 168
                         bins_per_octave=12)) # Standard 12-tone scale

    # 2. Power to Decibels
    arr2D = librosa.amplitude_to_db(C, ref=np.max)

    # 3. Find 2D Peaks
    local_maxima = get_2D_peaks(arr2D, amp_min=amp_min)

    # 4. Generate Transformation-Invariant Triplets
    return generate_triplet_hashes(local_maxima, fan_value=fan_value)


def get_2D_peaks(arr2D: np.ndarray, amp_min: int = DEFAULT_AMP_MIN) -> List[Tuple[int, int]]:
    struct = generate_binary_structure(2, CONNECTIVITY_MASK)
    neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D
    background = (arr2D == -80) # Librosa DB floor is usually -80
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks = local_max ^ eroded_background

    freqs, times = np.where(detected_peaks)
    amps = arr2D[detected_peaks]
    
    # Filter by amplitude
    filter_idxs = np.where(amps > amp_min)
    return sorted(zip(freqs[filter_idxs], times[filter_idxs]), key=itemgetter(1))

def generate_triplet_hashes(peaks, fan_value=DEFAULT_FAN_VALUE):
    """
    Vectorized version of triplet hashing.
    peaks: List of (f, t) tuples or a NumPy array of shape (N, 2)
    """
    peaks = np.array(peaks)
    n = len(peaks)
    if n < 3:
        return []

    f = peaks[:, 0]
    t = peaks[:, 1]
    
    all_hashes = []

    # Iterate through offsets j and k within the fan_value range
    # Since fan_value is small, this 'semi-vectorized' approach is 
    # significantly faster than a triple nested loop.
    for j in range(1, fan_value):
        for k in range(j + 1, fan_value + 1):
            # Indices for triplets: Anchor (i), Point 2 (i+j), Point 3 (i+k)
            # We slice the arrays so they align perfectly
            idx_i = np.arange(0, n - k)
            idx_j = idx_i + j
            idx_k = idx_i + k

            # Frequency components
            f1, f2, f3 = f[idx_i], f[idx_j], f[idx_k]
            # Time components
            t1, t2, t3 = t[idx_i], t[idx_j], t[idx_k]

            # Calculate Deltas
            dt21 = t2 - t1
            dt31 = t3 - t1
            df21 = f2 - f1
            df31 = f3 - f1

            # Mask to avoid division by zero and handle valid time deltas
            mask = dt31 != 0
            if not np.any(mask):
                continue

            # Calculate Invariants
            t_ratio = dt21[mask] / dt31[mask]
            v_df21 = df21[mask]
            v_df31 = df31[mask]
            v_t1 = t1[mask]

            # Vectorized Hash Generation
            # We use a vectorized string formatting approach
            for tr, d21, d31, time in zip(t_ratio, v_df21, v_df31, v_t1):
                h_str = f"{tr:.3f}|{d21}|{d31}"
                h = hashlib.sha1(h_str.encode("utf-8")).hexdigest()[:FINGERPRINT_REDUCTION]
                all_hashes.append((h, int(time)))

    return all_hashes
