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

def generate_triplet_hashes(peaks: List[Tuple[int, int]], fan_value: int = DEFAULT_FAN_VALUE):
    """
    Triplet Logic:
    Instead of (f1, f2, dt), we use (time_ratio, f2-f1, f3-f1).
    """
    n = len(peaks)
    hashes = []
    
    for i in range(n):
        # We look forward for two more points to form a triplet
        for j in range(1, fan_value):
            if (i + j) < n:
                for k in range(j + 1, fan_value + 1):
                    if (i + k) < n:
                        p1 = peaks[i]   # Anchor
                        p2 = peaks[i+j] # Point 2
                        p3 = peaks[i+k] # Point 3

                        # f = frequency bin index, t = time frame index
                        f1, t1 = p1
                        f2, t2 = p2
                        f3, t3 = p3

                        dt21 = t2 - t1
                        dt31 = t3 - t1
                        
                        if dt31 == 0: continue # Prevent division by zero

                        # --- THE INVARIANTS ---
                        # 1. Time Ratio: Invariant to speed changes
                        t_ratio = dt21 / dt31
                        
                        # 2. Frequency Deltas: Invariant to pitch shifts in CQT
                        df21 = f2 - f1
                        df31 = f3 - f1

                        # Generate Hash
                        h_str = f"{t_ratio:.3f}|{df21}|{df31}"
                        h = hashlib.sha1(h_str.encode("utf-8")).hexdigest()[:FINGERPRINT_REDUCTION]
                        
                        hashes.append((h, int(t1)))
                        
    return hashes
