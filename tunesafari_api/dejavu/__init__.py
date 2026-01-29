import multiprocessing
import os
import sys
import traceback
import hashlib
from itertools import groupby
from time import time
from datetime import datetime
from typing import Dict, List, Tuple
import librosa
import numpy as np
from collections import defaultdict
from scipy.ndimage import maximum_filter, binary_erosion
import dejavu.logic.decoder as decoder
from dejavu.base_classes.base_database import get_database
from dejavu.config.settings import (DEFAULT_FS, DEFAULT_OVERLAP_RATIO,
                                    DEFAULT_WINDOW_SIZE, FIELD_BLOB_SHA1,
                                    FIELD_TOTAL_HASHES,
                                    FINGERPRINTED_CONFIDENCE, PEAK_NEIGHBORHOOD_SIZE,
                                    FINGERPRINTED_HASHES, HASHES_MATCHED,
                                    INPUT_CONFIDENCE, INPUT_HASHES, OFFSET,
                                    OFFSET_SECS, SONG_ID, SONG_NAME, TOPN, DEFAULT_FAN_VALUE, DEFAULT_AMP_MIN, BUCKET_SIZE, N_BINS, BINS_PER_OCTAVE, HOP_LENGTH, MIN_NOTE,
                                    MAX_TIME_DELTA, MIN_TIME_DELTA, DETECTED_TEMPO)
from dejavu.logic.fingerprint import fingerprint

class Dejavu:
    def __init__(self, config):
        self.config = config

        # initialize db
        db_cls = get_database(config.get("database_type", "mysql").lower())
        redis_db_index = config.get("redis_db_index")
        self.db = db_cls(redis_db_index=redis_db_index, **config.get("database", {}))
        self.db.setup()

    def get_fingerprinted_songs(self) -> List[Dict[str, any]]:
        """
        To pull all fingerprinted songs from the database.

        :return: a list of fingerprinted audios from the database.
        """
        return self.db.get_songs()

    def delete_songs_by_id(self, song_ids: List[int]) -> None:
        """
        Deletes all audios given their ids.

        :param song_ids: song ids to delete from the database.
        """
        self.db.delete_songs_by_id(song_ids)

    def fingerprint_blob(self, blob, song_name: str = None, remote_addr=None) -> Tuple[int, str]:
        if not song_name: 
            return -1, None
        
        try:
            # 1. Decode the audio binary
            channels, fs, file_hash = decoder.read(blob)
            
            all_hashes = []
            # 2. Generate fingerprints for each channel
            for channel in channels:
                # Call your instance method that uses CQT and Triplets
                hashes, _ = self.generate_fingerprints(channel, Fs=fs)
                all_hashes.extend(hashes)
            
            if len(all_hashes) == 0:
                print("WARNING: No hashes generated! Check your amp_min threshold.")
                return 3, file_hash

            # 3. Insert into Database
            # Note: all_hashes is a list of (hash_string, offset)
            sid = self.db.insert_song(song_name, file_hash, len(all_hashes))
            self.db.insert_hashes(sid, all_hashes)
            self.db.set_song_fingerprinted(sid)
            
            return 0, file_hash
            
        except Exception as e:
            sys.stderr.write(f"CRITICAL ERROR: {e}\n")
            traceback.print_exc()
            return 2, None

    def generate_fingerprints(self, samples: np.ndarray, Fs=DEFAULT_FS) -> Tuple[List[Tuple[str, int]], float]:
        t = time()
        samples_float = samples.astype(float)

        # CQT
        cqt = np.abs(librosa.cqt(samples_float, sr=Fs, hop_length=HOP_LENGTH, 
                                 fmin=librosa.note_to_hz(MIN_NOTE), 
                                 n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE))
        
        # This makes the loudest peak 0dB and the rest negative.
        spectrogram_db = librosa.amplitude_to_db(cqt, ref=np.max)
        
        # Handle noise floor
        peaks = self._extract_peaks(spectrogram_db, threshold=DEFAULT_AMP_MIN)
        
        # 4. Hash Triplets
        hashes = self._generate_hashes(peaks)
        
        return hashes, time() - t

    def _extract_peaks(self, spectrogram, threshold=DEFAULT_AMP_MIN) -> List[Tuple[int, int]]:
        peak_nei_size = PEAK_NEIGHBORHOOD_SIZE
        
        # Find Local Maxima using the neighborhood size
        # size=(y, x) -> (frequency_bins, time_frames)
        local_max = maximum_filter(spectrogram, size=(peak_nei_size, peak_nei_size)) == spectrogram
        
        # Vectorized Amplitude Filter (Faster than a for-loop)
        # This replaces the background/erosion logic with a simple, reliable threshold
        amps = spectrogram[local_max]
        freqs, times = np.where(local_max)
        
        # Filter indices where amplitude is above threshold
        filter_idxs = np.where(amps > threshold)
        
        # Create the list of (freq, time) tuples
        peak_list = list(zip(freqs[filter_idxs], times[filter_idxs]))
        
        return sorted(peak_list, key=lambda x: x[1])

    def _generate_hashes(self, peaks, fan_out=DEFAULT_FAN_VALUE) -> List[Tuple[int, int]]:
        peaks = np.array(peaks)
        n = len(peaks)
        if n < 3: return []

        f = peaks[:, 0].astype(np.int32)
        t = peaks[:, 1]

        # Generate all (j, k) index pairs relative to each anchor
        # Instead of nested loops, we create a meshgrid of neighbor offsets
        j_offsets, k_offsets = np.triu_indices(fan_out + 1, k=1)
        # triu_indices gives us pairs like (0,1), (0,2)...(1,2), (1,3)...
        # Since your original loop started at j=1, we shift these to match your logic
        j_offsets += 1
        k_offsets += 1

        # Create indices for all anchors and their corresponding j,k neighbors
        # This creates a matrix of shape (N, num_triplets)
        anchors = np.arange(n).reshape(-1, 1)
        idx_j = anchors + j_offsets
        idx_k = anchors + k_offsets

        # Filter out-of-bounds indices
        # This replaces 'idx_limit = n - k'
        valid_idx_mask = idx_k < n
        
        # Flatten these into long vectors for single-pass processing
        row_idx, col_idx = np.where(valid_idx_mask)
        target_j = idx_j[row_idx, col_idx]
        target_k = idx_k[row_idx, col_idx]
        target_a = row_idx # The anchor index

        # Pull data using fancy indexing (The "Big Vectorization" step)
        f1, f2, f3 = f[target_a], f[target_j], f[target_k]
        t1, t2, t3 = t[target_a], t[target_j], t[target_k]

        # Calculate Deltas and enforce your dt31 > 0 constant
        dt21 = t2 - t1
        dt31 = t3 - t1
        
        # Filter valid masks
        valid_mask = (dt21 >= MIN_TIME_DELTA) & (dt21 <= MAX_TIME_DELTA) & (dt31 >= MIN_TIME_DELTA) & (dt31 <= MAX_TIME_DELTA) & (dt31 > dt21)
        
        # Final Logic & Packing (Vectorized)
        t_ratios = (dt21[valid_mask] / dt31[valid_mask] * 1000).astype(np.int32)
        df21 = (f2[valid_mask] - f1[valid_mask])
        df31 = (f3[valid_mask] - f1[valid_mask])
        anchor_times = t1[valid_mask].astype(np.int32)

        # bit-packing logic
        packed_hashes = ((t_ratios & 0xFFFF) << 16) | \
                        (((df21 + 128) & 0xFF) << 8) | \
                        ((df31 + 128) & 0xFF)

        # Return as list of tuples
        return list(zip(packed_hashes.tolist(), anchor_times.tolist()))

    def find_matches(self, hashes: List[Tuple[int, int]]) -> Tuple[List[Tuple[any, int, int]], Dict[any, int], float]:
        """
        Finds the corresponding matches on the fingerprinted audios for the given hashes.

        :param hashes: list of tuples for hashes and their corresponding offsets (from query)
        :return: 
            - matches: List of (song_id, db_offset, query_offset)
            - dedup_hashes: Dictionary of {song_id: total_match_count}
            - query_time: Float representing seconds taken
        """
        t = time()
        # self.db.return_matches now returns the 3-tuple (sid, db_off, q_off)
        # plus the dedup_hashes dictionary.
        matches, dedup_hashes = self.db.return_matches(hashes)
        query_time = time() - t
        return matches, dedup_hashes, query_time

    def align_matches(self, matches: List[Tuple[int, int, int]], queried_hashes: int,
                  topn: int = TOPN) -> List[Dict[str, any]]:
        if not matches: return []

        # 1. Faster grouping
        song_match_pairs = defaultdict(list)
        for sid, db_off, q_off in matches:
            song_match_pairs[sid].append((db_off, q_off))

        tempo_scales = np.linspace(0.8, 1.5, 36)
        scales_grid = tempo_scales[:, np.newaxis] 
        
        songs_matches = []

        for song_id, pairs in song_match_pairs.items():
            # Candidate Filter: Keep at 15 to filter out extreme low-level noise
            if len(pairs) < 15: 
                continue
                
            pairs_np = np.array(pairs)
            db_offs = pairs_np[:, 0]
            q_offs = pairs_np[:, 1]
            
            # 2. VECTORIZED MATH: Calculate all tempos at once
            all_diffs = db_offs - (q_offs * scales_grid)
            
            best_count = 0
            best_off = 0
            best_t = 1.0

            for i, s in enumerate(tempo_scales):
                diffs = all_diffs[i]
                
                # Use BUCKET_SIZE constant
                shifted = (diffs // BUCKET_SIZE).astype(int)
                min_val = shifted.min()
                counts = np.bincount(shifted - min_val)
                
                if len(counts) == 0: continue

                # SLIDING WINDOW: This captures the signal if it's split between two buckets
                # Crucial for huge DBs where noise is high
                if len(counts) > 1:
                    combined_counts = counts[:-1] + counts[1:]
                    current_max = combined_counts.max()
                    max_idx = combined_counts.argmax()
                else:
                    current_max = counts[0]
                    max_idx = 0
                
                if current_max > best_count:
                    best_count = current_max
                    # Use max_idx as the base for the offset
                    best_off = (max_idx + min_val) * BUCKET_SIZE
                    best_t = s

            # --- THE MATHEMATICAL FIX ---
            # best_count is the "Signal" (clustered matches)
            # len(pairs) is the "Total Noise" for this song
            # Cubing best_count ensures that high-density alignment is rewarded 
            # much more than high-volume random collisions.
            final_score = (best_count ** 3) / len(pairs)

            songs_matches.append((song_id, best_off, best_count, best_t, final_score))

        # Sort by the final_score (index 4) instead of raw count
        songs_matches.sort(key=lambda x: x[4], reverse=True)

        # Build JSON output structure
        songs_result = []
        for song_id, offset, hashes_matched_count, detected_tempo, score in songs_matches[0:topn]:
            song = self.db.get_song_by_id(song_id)
            if not song: continue

            raw_sha1 = song.get(FIELD_BLOB_SHA1)
            blob_sha1_str = raw_sha1.hex() if isinstance(raw_sha1, bytes) else str(raw_sha1)

            # Constant: HOP_LENGTH / DEFAULT_FS
            nseconds = round(float(offset) * HOP_LENGTH / DEFAULT_FS, 5)
            total_hashes_in_db = song.get(FIELD_TOTAL_HASHES) or 1

            songs_result.append({
                SONG_ID: str(song_id),
                SONG_NAME: song.get(SONG_NAME),
                INPUT_HASHES: queried_hashes,
                FINGERPRINTED_HASHES: total_hashes_in_db,
                HASHES_MATCHED: hashes_matched_count,
                #"SCORE": round(score, 2), 
                INPUT_CONFIDENCE: hashes_matched_count / queried_hashes,
                FINGERPRINTED_CONFIDENCE: hashes_matched_count / total_hashes_in_db,
                DETECTED_TEMPO: round(detected_tempo, 2),
                OFFSET: max(0, int(offset)),
                OFFSET_SECS: max(0, nseconds),
                FIELD_BLOB_SHA1: blob_sha1_str.lower()
            })

        return songs_result

    def recognize(self, recognizer, *options, **kwoptions) -> Dict[str, any]:
        r = recognizer(self)
        return r.recognize(*options, **kwoptions)

    @staticmethod
    def _fingerprint_worker(blob, song_name, remote_addr):
        # Use the decoder to get the audio data
        channels, fs, file_hash = decoder.read(blob)
        
        all_hashes = []
        for channel in channels:
            # Note: You should move the CQT logic to a standalone function 
            # in logic/fingerprint.py so this worker can call it easily.
            hashes = fingerprint(channel, Fs=fs) 
            all_hashes.extend(hashes)
            
        return all_hashes, file_hash

    @staticmethod
    def get_blob_fingerprints(blob, song_name, remote_addr, print_output: bool = True):
        channels, fs, file_hash = decoder.read(blob)
        fingerprints = set()
        channel_amount = len(channels)
        for channeln, channel in enumerate(channels, start=1):
            if print_output:
                print(f"{datetime.now().strftime("[%d/%b/%Y %H:%M:%S]")} {remote_addr} \"INFO: Fingerprinting channel {channeln}/{channel_amount} for {song_name}, blob_sha1: {file_hash.lower()}\"")

            hashes = fingerprint(channel, Fs=fs)

            if print_output:
                print(f"{datetime.now().strftime("[%d/%b/%Y %H:%M:%S]")} {remote_addr} \"INFO: Finished channel {channeln}/{channel_amount} for {song_name}, blob_sha1: {file_hash.lower()}\"")

            fingerprints |= set(hashes)

        return fingerprints, file_hash.lower()
