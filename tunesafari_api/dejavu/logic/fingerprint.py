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
    struct = generate_binary_structure(2, CONNECTIVITY_MASK)
    neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D
    background = (arr2D > DEFAULT_AMP_MIN) # Background noise floor
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks = local_max ^ eroded_background

    freqs, times = np.where(detected_peaks)
    amps = arr2D[detected_peaks]
    
    # Filter by amplitude
    filter_idxs = np.where(amps > amp_min)
    return sorted(zip(freqs[filter_idxs], times[filter_idxs]), key=itemgetter(1))

def generate_triplet_hashes(peaks, fan_value=DEFAULT_FAN_VALUE, 
                            min_delta_t=MIN_TIME_DELTA, max_delta_t=MAX_TIME_DELTA):
    peaks = np.array(peaks)
    n = len(peaks)
    if n < 3:
        return []

    f = peaks[:, 0].astype(np.int32)
    t = peaks[:, 1]

    # 1. Broad-stroke neighbor discovery
    # Create a distance matrix for time: (N, fan_value + 1)
    # We look at the next 'fan_value + offset' points to find valid window candidates
    max_search = fan_value * 2 
    idx = np.arange(n)
    
    # Efficiently gather neighbors using a sliding window view
    # padding ensures we don't out-of-bounds
    padded_t = np.pad(t, (0, max_search), constant_values=np.inf)
    padded_f = np.pad(f, (0, max_search), constant_values=0)
    
    all_hashes = []

    for i in range(n):
        # Look ahead at a block of potential candidates
        t_neighbors = padded_t[i+1 : i + max_search + 1]
        f_neighbors = padded_f[i+1 : i + max_search + 1]
        
        # 2. Apply Time Window Mask
        # Find indices within [min_delta_t, max_delta_t]
        dt = t_neighbors - t[i]
        mask = (dt >= min_delta_t) & (dt <= max_delta_t)
        
        valid_indices = np.where(mask)[0][:fan_value]
        num_valid = len(valid_indices)
        
        if num_valid < 2:
            continue

        # Extract only valid candidate data
        v_dt = dt[valid_indices]
        v_f = f_neighbors[valid_indices]
        v_df = v_f - f[i]

        # 3. Vectorized Triplet Formation for Anchor i
        # We generate all pairs (j, k) from the valid candidates
        # Combinations: j from 0 to num_valid-1, k from j+1 to num_valid
        for j in range(num_valid):
            # Point 2 data
            dt21 = v_dt[j]
            df21 = v_df[j]
            
            # Points 3 data (all points after j)
            dt31 = v_dt[j+1:]
            df31 = v_df[j+1:]
            
            # Vectorized calculation of ratios and hashes for all k in this j
            t_ratios = (dt21 / dt31 * 100).astype(np.int32)
            
            # Bit-pack the hashes
            # Packs: Ratio (high bits), df21, df31 (low bits)
            hashes = (t_ratios << 16) | ((df21 + 100) << 8) | (df31 + 100)
            
            # Append as (hash, anchor_time)
            anchor_time = int(t[i])
            all_hashes.extend([(int(h), anchor_time) for h in hashes])

    return all_hashes
