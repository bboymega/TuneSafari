import hashlib
from operator import itemgetter
from typing import List, Tuple

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (binary_erosion,
                                      generate_binary_structure,
                                      iterate_structure)

from dejavu.config.settings import (CONNECTIVITY_MASK, DEFAULT_AMP_MIN,
                                    DEFAULT_FAN_VALUE, DEFAULT_FS,
                                    DEFAULT_OVERLAP_RATIO, DEFAULT_WINDOW_SIZE, MAX_HASH_TIME_DELTA,
                                    MIN_HASH_TIME_DELTA,
                                    PEAK_NEIGHBORHOOD_SIZE, PEAK_SORT)


def fingerprint(channel_samples: List[int],
                Fs: int = DEFAULT_FS,
                wsize: int = DEFAULT_WINDOW_SIZE,
                wratio: float = DEFAULT_OVERLAP_RATIO,
                fan_value: int = DEFAULT_FAN_VALUE,
                amp_min: int = DEFAULT_AMP_MIN) -> List[Tuple[str, int]]:
    """
    FFT the channel, log transform output, find local maxima, then return locally sensitive hashes.

    :param channel_samples: channel samples to fingerprint.
    :param Fs: audio sampling rate.
    :param wsize: FFT windows size.
    :param wratio: ratio by which each sequential window overlaps the last and the next window.
    :param fan_value: degree to which a fingerprint can be paired with its neighbors.
    :param amp_min: minimum amplitude in spectrogram in order to be considered a peak.
    :return: a list of hashes with their corresponding offsets.
    """
    # FFT the signal and extract frequency components
    arr2D = mlab.specgram(
        channel_samples,
        NFFT=wsize,
        Fs=Fs,
        window=mlab.window_hanning,
        noverlap=int(wsize * wratio))[0]

    # Apply log transform since specgram function returns linear array. 0s are excluded to avoid np warning.
    arr2D = 10 * np.log10(arr2D, out=np.zeros_like(arr2D), where=(arr2D != 0))

    local_maxima = get_2D_peaks(arr2D, plot=False, amp_min=amp_min)

    # return hashes
    return generate_hashes(local_maxima, fan_value=fan_value)


def get_2D_peaks(arr2D: np.array, plot: bool = False, amp_min: int = DEFAULT_AMP_MIN)\
        -> List[Tuple[List[int], List[int]]]:
    """
    Extract maximum peaks from the spectogram matrix (arr2D).

    :param arr2D: matrix representing the spectogram.
    :param plot: for plotting the results.
    :param amp_min: minimum amplitude in spectrogram in order to be considered a peak.
    :return: a list composed by a list of frequencies and times.
    """
    # Original code from the repo is using a morphology mask that does not consider diagonal elements
    # as neighbors (basically a diamond figure) and then applies a dilation over it, so what I'm proposing
    # is to change from the current diamond figure to a just a normal square one:
    #       F   T   F           T   T   T
    #       T   T   T   ==>     T   T   T
    #       F   T   F           T   T   T
    # In my local tests time performance of the square mask was ~3 times faster
    # respect to the diamond one, without hurting accuracy of the predictions.
    # I've made now the mask shape configurable in order to allow both ways of find maximum peaks.
    # That being said, we generate the mask by using the following function
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generate_binary_structure.html
    struct = generate_binary_structure(2, CONNECTIVITY_MASK)

    #  And then we apply dilation using the following function
    #  http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.iterate_structure.html
    #  Take into account that if PEAK_NEIGHBORHOOD_SIZE is 2 you can avoid the use of the scipy functions and just
    #  change it by the following code:
    #  neighborhood = np.ones((PEAK_NEIGHBORHOOD_SIZE * 2 + 1, PEAK_NEIGHBORHOOD_SIZE * 2 + 1), dtype=bool)
    neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    # find local maxima using our filter mask
    local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D

    # Applying erosion, the dejavu documentation does not talk about this step.
    background = (arr2D == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # Boolean mask of arr2D with True at peaks (applying XOR on both matrices).
    detected_peaks = local_max != eroded_background

    # extract peaks
    amps = arr2D[detected_peaks]
    freqs, times = np.where(detected_peaks)

    # filter peaks
    amps = amps.flatten()

    # get indices for frequency and time
    filter_idxs = np.where(amps > amp_min)

    freqs_filter = freqs[filter_idxs]
    times_filter = times[filter_idxs]

    if plot:
        # scatter of the peaks
        fig, ax = plt.subplots()
        ax.imshow(arr2D)
        ax.scatter(times_filter, freqs_filter)
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        ax.set_title("Spectrogram")
        plt.gca().invert_yaxis()
        plt.show()

    return list(zip(freqs_filter, times_filter))
    

def generate_hashes(peaks, fan_value=DEFAULT_FAN_VALUE):
    peaks = np.asarray(peaks)
    if PEAK_SORT:
        peaks = peaks[np.argsort(peaks[:, 1], kind='stable')]

    freqs = peaks[:, 0].astype(np.uint64)
    times = peaks[:, 1].astype(np.uint64)
    n = len(peaks)

    # 1. Generate indices and offsets
    i_idx = np.arange(n).reshape(n, 1)
    j_offsets = np.arange(1, fan_value)
    j_idx = i_idx + j_offsets

    # 2. Create and apply the boundary mask
    valid = j_idx < n
    i_idx_expanded = np.broadcast_to(i_idx, j_idx.shape)
    
    i_v = i_idx_expanded[valid]
    j_v = j_idx[valid]
    
    # 3. Time Delta constraints
    t1, t2 = times[i_v], times[j_v]
    t_delta = t2 - t1
    mask = (t_delta >= MIN_HASH_TIME_DELTA) & (t_delta <= MAX_HASH_TIME_DELTA)
    
    # 4. Final selection and Packing
    f1 = freqs[i_v[mask]]
    f2 = freqs[j_v[mask]]
    dt = t_delta[mask].astype(np.uint64)
    t1_final = t1[mask].astype(np.uint32)

    # Pack: [F1: 20 bits] [F2: 20 bits] [Delta: 24 bits]
    packed_hashes = (f1 << 44) | (f2 << 24) | dt

    return list(zip(packed_hashes, t1_final))
