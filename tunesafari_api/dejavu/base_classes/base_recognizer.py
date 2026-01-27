import abc
from time import time
from typing import Dict, List, Tuple

import numpy as np

from dejavu.config.settings import DEFAULT_FS


class BaseRecognizer(object, metaclass=abc.ABCMeta):
    def __init__(self, dejavu):
        self.dejavu = dejavu
        self.Fs = DEFAULT_FS

    def _recognize(self, *data) -> Tuple[List[Dict[str, any]], int, int, int]:
        fingerprint_times = []
        hashes = set()  # to remove possible duplicated fingerprints
        
        for channel in data:
            fingerprints, fingerprint_time = self.dejavu.generate_fingerprints(channel, Fs=self.Fs)
            fingerprint_times.append(fingerprint_time)
            hashes |= set(fingerprints)

        # 1. matches now contains (song_id, db_offset, query_offset)
        matches, dedup_hashes, query_time = self.dejavu.find_matches(list(hashes))

        t = time()
        # 2. FIX: Remove 'dedup_hashes' from this call. 
        # align_matches(matches, queried_hashes_count)
        final_results = self.dejavu.align_matches(matches, len(hashes))
        align_time = time() - t

        return final_results, np.sum(fingerprint_times), query_time, align_time
