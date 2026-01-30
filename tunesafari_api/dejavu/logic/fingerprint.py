import hashlib
from operator import itemgetter
from typing import List, Tuple
import librosa
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import maximum_filter
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage.morphology import (binary_erosion,
                                      generate_binary_structure,
                                      iterate_structure)

from dejavu.config.settings import (CONNECTIVITY_MASK, DEFAULT_AMP_MIN,
                                    DEFAULT_FAN_VALUE, DEFAULT_FS,
                                    DEFAULT_OVERLAP_RATIO, DEFAULT_WINDOW_SIZE,
                                    FINGERPRINT_REDUCTION, MAX_HASH_TIME_DELTA,
                                    MIN_HASH_TIME_DELTA,
                                    PEAK_NEIGHBORHOOD_SIZE, PEAK_SORT, MIN_TIME_DELTA, MAX_TIME_DELTA, N_BINS, BINS_PER_OCTAVE, HOP_LENGTH, MIN_NOTE, DEFAULT_AMP_MIN)


def fingerprint(channel_samples: np.ndarray,
                Fs: int = DEFAULT_FS,
                fan_value: int = DEFAULT_FAN_VALUE,
                amp_min: int = DEFAULT_AMP_MIN) -> List[Tuple[str, int]]:
    """
    Uses CQT for pitch invariance and Triplets for time invariance.
    """

    C = np.abs(librosa.cqt(channel_samples.astype(float), 
                         sr=Fs, 
                         hop_length=HOP_LENGTH, 
                         fmin=librosa.note_to_hz(MIN_NOTE), 
                         n_bins=N_BINS, 
                         bins_per_octave=BINS_PER_OCTAVE)) # Default: Standard 12-tone scale

    # Power to Decibels
    arr2D = librosa.amplitude_to_db(C, ref=np.max)

    # Find 2D Peaks
    local_maxima = get_2D_peaks(arr2D, amp_min=amp_min)

    # Generate Transformation-Invariant Triplets
    return generate_triplet_hashes(local_maxima, fan_value=fan_value)


def get_2D_peaks(arr2D: np.ndarray, amp_min: int = DEFAULT_AMP_MIN) -> List[Tuple[int, int]]:
    # 1. Define the connectivity (usually 1 or 2)
    struct = generate_binary_structure(2, CONNECTIVITY_MASK)
    neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    # Find local maxima
    local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D
    
    # Use a simple amplitude mask instead of erosion logic.
    # This ensures we don't accidentally delete peaks in loud musical sections.
    amps = arr2D[local_max]
    freqs, times = np.where(local_max)

    # Filter by amplitude
    filter_idxs = np.where(amps > amp_min)
    
    # Return sorted by time
    return sorted(zip(freqs[filter_idxs], times[filter_idxs]), key=itemgetter(1))


def generate_triplet_hashes(peaks, fan_value=DEFAULT_FAN_VALUE, 
                            min_delta_t=MIN_TIME_DELTA, max_delta_t=MAX_TIME_DELTA):
    peaks = np.array(peaks)
    n = len(peaks)
    if n < 3: return []

    f = peaks[:, 0].astype(np.uint64)
    t = peaks[:, 1]

    # Increase lookahead locally to ensure we find triplets in dense databases
    # 40-60 is better for large databases
    lookahead = fan_value * 3 
    
    f_padded = np.pad(f, (0, lookahead), constant_values=0)
    t_padded = np.pad(t, (0, lookahead), constant_values=np.inf)

    f_neighbors = sliding_window_view(f_padded[1:], window_shape=lookahead)[:n]
    t_neighbors = sliding_window_view(t_padded[1:], window_shape=lookahead)[:n]

    dt_matrix = t_neighbors - t[:, np.newaxis]
    df_matrix = f_neighbors - f[:, np.newaxis]

    all_hashes = []
    all_anchors = []

    for j in range(lookahead):
        for k in range(j + 1, lookahead):
            dt21 = dt_matrix[:, j]
            dt31 = dt_matrix[:, k]
            
            # We allow a slightly larger time window locally to catch 1.4x changes
            mask = (dt21 >= min_delta_t) & (dt31 <= (max_delta_t * 1.5)) & (dt31 > dt21)
            
            if not np.any(mask):
                continue

            v_dt21 = dt21[mask]
            v_dt31 = dt31[mask]
            
            # 10 bits for Ratio (0-1023)
            t_ratios = (v_dt21 / v_dt31 * 1024).astype(np.uint64)

            v_df21 = df_matrix[mask, j]
            v_df31 = df_matrix[mask, k]
            
            # PACKING FOR HUGE DATABASES:
            # We use 32-bit integers: 10 bits ratio, 11 bits df21, 11 bits df31.
            # Adding 512 handles a wider range of pitch/frequency shifts.
            h = ((t_ratios & 0x3FF) << 22) | \
                (((v_df21 + 512) & 0x7FF) << 11) | \
                ((v_df31 + 512) & 0x7FF)

            all_hashes.append(h)
            all_anchors.append(t[mask].astype(np.uint64))

    if not all_hashes:
        return []

    res_h = np.concatenate(all_hashes)
    res_t = np.concatenate(all_anchors)

    return list(zip(res_h.tolist(), res_t.tolist()))