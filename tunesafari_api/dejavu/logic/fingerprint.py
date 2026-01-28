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

    # Keep your exact types
    f = peaks[:, 0].astype(np.int32)
    t = peaks[:, 1]

    # We use a fixed lookahead window to allow vectorization. 
    # To ensure we don't miss peaks within max_delta_t, we use a safe buffer.
    lookahead = fan_value * 3 
    
    # Pad and create sliding window views for all neighbors at once
    f_padded = np.pad(f, (0, lookahead), constant_values=0)
    t_padded = np.pad(t, (0, lookahead), constant_values=np.inf)

    # These matrices are (N, lookahead)
    # Each row i contains the neighbors for anchor i
    f_neighbors = sliding_window_view(f_padded[1:], window_shape=lookahead)[:n]
    t_neighbors = sliding_window_view(t_padded[1:], window_shape=lookahead)[:n]

    # Vectorized Delta Calculation (Broadcasting)
    # t[:, None] reshapes t to (N, 1) so it broadcasts against (N, lookahead)
    dt_matrix = t_neighbors - t[:, np.newaxis]
    df_matrix = f_neighbors - f[:, np.newaxis]

    # Create a Validity Mask
    # This ensures neighbors are within your MIN/MAX constants
    # 1. Ensure Point 2 is within bounds of Anchor 
    # 2. Ensure Point 3 is within bounds of Anchor
    # 3. Ensure Point 3 is strictly AFTER Point 2 (dt32 > 0)
    m2 = (dt_matrix[:, j] >= min_delta_t) & (dt_matrix[:, j] <= max_delta_t)
    m3 = (dt_matrix[:, k] >= min_delta_t) & (dt_matrix[:, k] <= max_delta_t)
    m32 = (dt_matrix[:, k] > dt_matrix[:, j])

    valid_mask = m2 & m3 & m32

    all_hashes = []
    all_anchors = []

    # Generate Triplets
    # Instead of looping over N (songs), we loop over the lookahead depth
    # This is much faster because fan_value is small (e.g., 20)
    for j in range(lookahead):
        # Point 2 candidates
        dt21 = dt_matrix[:, j]
        df21 = df_matrix[:, j]
        m2 = valid_mask[:, j]

        for k in range(j + 1, lookahead):
            # Point 3 candidates
            dt31 = dt_matrix[:, k]
            df31 = df_matrix[:, k]
            m3 = valid_mask[:, k]

            # Only process anchors where both neighbor j and k are valid
            combined_mask = m2 & m3
            if not np.any(combined_mask):
                continue

            # CONSTANTS & LOGIC:
            # t_ratios calculation
            v_dt21 = dt21[combined_mask]
            v_dt31 = dt31[combined_mask]
            t_ratios = (v_dt21 / v_dt31 * 1000).astype(np.int32)

            # PACKING LOGIC:
            v_df21 = df21[combined_mask]
            v_df31 = df31[combined_mask]
            
            # Applying bit-shifts and masks to prevent overflow
            h = ((t_ratios & 0xFFFF) << 16) | \
                (((v_df21 + 128) & 0xFF) << 8) | \
                ((v_df31 + 128) & 0xFF)

            all_hashes.append(h)
            all_anchors.append(t[combined_mask].astype(np.int32))

    if not all_hashes:
        return []

    # Final flattening of vectorized blocks
    res_h = np.concatenate(all_hashes)
    res_t = np.concatenate(all_anchors)

    return list(zip(res_h.tolist(), res_t.tolist()))
