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

        f = peaks[:, 0].astype(np.uint64)
        t = peaks[:, 1]

        # 1. INCREASE SEARCH DENSITY
        # To handle 1.4x speed, we need to look at more neighbors because some 
        # original neighbors might now be too close or filtered out.
        local_fan = fan_out + 10 
        j_offsets, k_offsets = np.triu_indices(local_fan, k=1)
        j_offsets += 1
        k_offsets += 1

        anchors = np.arange(n).reshape(-1, 1)
        idx_j = anchors + j_offsets
        idx_k = anchors + k_offsets

        valid_idx_mask = idx_k < n
        row_idx, col_idx = np.where(valid_idx_mask)
        
        target_a = row_idx 
        target_j = idx_j[row_idx, col_idx]
        target_k = idx_k[row_idx, col_idx]

        f1, f2, f3 = f[target_a], f[target_j], f[target_k]
        t1, t2, t3 = t[target_a], t[target_j], t[target_k]

        dt21 = t2 - t1
        dt31 = t3 - t1
        
        # 2. WIDEN THE TIME WINDOW
        # A 1.4x speedup means the original gaps were 40% larger.
        # We broaden the max_delta locally so the compressed input still finds the database.
        local_max_delta = MAX_TIME_DELTA * 2 
        valid_mask = (dt21 >= MIN_TIME_DELTA) & (dt31 <= local_max_delta) & (dt31 > dt21)
        
        # 3. HIGH-PRECISION RATIO & PACKING
        # Using 1024 (2^10) instead of 1000 makes the ratio more bit-friendly.
        t_ratios = (dt21[valid_mask] / dt31[valid_mask] * 1024).astype(np.uint64)
        df21 = (f2[valid_mask] - f1[valid_mask])
        df31 = (f3[valid_mask] - f1[valid_mask])
        anchor_times = t1[valid_mask].astype(np.uint64)

        # 4. LARGE DATABASE BIT-PACKING (32-bit total)
        # 10 bits for ratio (0-1023)
        # 11 bits for df21 (allows difference of +/- 1024)
        # 11 bits for df31 (allows difference of +/- 1024)
        # This reduces collisions by 16x compared to your 8-bit logic.
        packed_hashes = ((t_ratios & 0x3FF) << 22) | \
                        (((df21 + 512) & 0x7FF) << 11) | \
                        ((df31 + 512) & 0x7FF)

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

        # 1. Group matches by Song ID
        song_match_pairs = defaultdict(list)
        for sid, db_off, q_off in matches:
            song_match_pairs[sid].append((db_off, q_off))

        # 2. HIGH-RESOLUTION TEMPO GRID
        tempo_scales = np.linspace(0.8, 1.5, 141)
        scales_grid = tempo_scales[:, np.newaxis] 
        
        candidate_songs = []

        for song_id, pairs in song_match_pairs.items():
            if len(pairs) < 15: continue
                
            pairs_np = np.array(pairs)
            db_offs = pairs_np[:, 0]
            q_offs = pairs_np[:, 1]
            
            all_diffs = db_offs - (q_offs * scales_grid)
            
            best_count = 0
            best_off = 0
            best_t = 1.0

            for i, s in enumerate(tempo_scales):
                diffs = all_diffs[i]
                # Use a more forgiving bucket for high-speed shifts to prevent peak splitting
                dynamic_bucket = BUCKET_SIZE * (1.0 + abs(1.0 - s) * 0.6)
                
                shifted = (diffs // dynamic_bucket).astype(int)
                min_val = shifted.min()
                
                if shifted.max() - min_val > 200000: continue
                
                counts = np.bincount(shifted - min_val)
                if len(counts) == 0: continue

                if len(counts) >= 3:
                    smoothed = np.convolve(counts, [0.5, 1, 0.5], mode='same')
                    current_max = smoothed.max()
                    max_idx = smoothed.argmax()
                else:
                    current_max = counts.max()
                    max_idx = counts.argmax()
                
                if current_max > best_count:
                    best_count = current_max
                    best_off = (max_idx + min_val) * dynamic_bucket
                    best_t = s

            # Calculate Query Coverage: How much of the input audio actually matched?
            unique_q_hits = len(np.unique(q_offs))
            
            # PRE-SCORE: Favors absolute match volume
            base_score = (best_count ** 2) * (best_count / (unique_q_hits + 1))
            
            candidate_songs.append({
                "song_id": song_id,
                "offset": best_off,
                "count": best_count,
                "tempo": best_t,
                "score": base_score,
                "pairs": pairs_np,
                "unique_q": unique_q_hits
            })

        # 3. DENSITY-WEIGHTED SNR VERIFICATION
        verified_results = []
        candidate_songs.sort(key=lambda x: x["score"], reverse=True)
        
        for candidate in candidate_songs[:20]:
            t = candidate["tempo"]
            off = candidate["offset"]
            p_np = candidate["pairs"]
            
            aligned = p_np[:, 0] - (p_np[:, 1] * t)
            
            # Strict alignment window
            peak_mask = np.abs(aligned - off) < (BUCKET_SIZE * 1.5)
            peak_count = np.sum(peak_mask)
            
            if peak_count < 12: continue
            
            tightness = np.std(aligned[peak_mask])
            
            # FINAL POWER SCORE
            # 1. Cubing peak_count makes the 545 vs 304 gap insurmountable.
            # 2. Multiplying by unique_q ensures the matches aren't just one looping sound.
            # 3. Tightness penalty remains to ensure it's a clean line.
            final_score = (peak_count ** 3 * candidate["unique_q"]) / (max(tightness, 0.001) + 0.8)
            
            candidate["score"] = final_score
            verified_results.append(candidate)

        verified_results.sort(key=lambda x: x["score"], reverse=True)

        # 4. FORMAT OUTPUT
        songs_result = []
        for res in verified_results[:topn]:
            sid = res["song_id"]
            song = self.db.get_song_by_id(sid)
            
            nseconds = round(float(res["offset"]) * HOP_LENGTH / DEFAULT_FS / res["tempo"], 5)
            
            # Relative confidence across the top results
            top_scores_sum = sum(c['score'] for c in verified_results[:5])
            relative_conf = (res["score"] / top_scores_sum) if top_scores_sum > 0 else 0
            
            songs_result.append({
                SONG_ID: str(sid),
                SONG_NAME: song.get(SONG_NAME),
                INPUT_HASHES: queried_hashes,
                FINGERPRINTED_HASHES: song.get(FIELD_TOTAL_HASHES) or 1,
                HASHES_MATCHED: int(res["count"]),
                INPUT_CONFIDENCE: round(res["count"] / queried_hashes, 5),
                FINGERPRINTED_CONFIDENCE: round(relative_conf, 5),
                DETECTED_TEMPO: round(res["tempo"], 2),
                OFFSET: max(0, int(res["offset"])),
                OFFSET_SECS: max(0, nseconds),
                FIELD_BLOB_SHA1: song.get(FIELD_BLOB_SHA1).hex().lower() if isinstance(song.get(FIELD_BLOB_SHA1), bytes) else str(song.get(FIELD_BLOB_SHA1)).lower()
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
