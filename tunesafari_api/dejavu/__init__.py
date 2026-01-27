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
                                    FINGERPRINTED_CONFIDENCE,
                                    FINGERPRINTED_HASHES, HASHES_MATCHED,
                                    INPUT_CONFIDENCE, INPUT_HASHES, OFFSET,
                                    OFFSET_SECS, SONG_ID, SONG_NAME, TOPN, DEFAULT_FAN_VALUE, DEFAULT_AMP_MIN, BUCKET_SIZE, N_BINS, BINS_PER_OCTAVE, HOP_LENGTH, MIN_NOTE)
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

        # 1. CQT - Set to 84 bins (7 octaves) to stay safely below Nyquist limit
        cqt = np.abs(librosa.cqt(samples_float, sr=Fs, hop_length=HOP_LENGTH, 
                                 fmin=librosa.note_to_hz(MIN_NOTE), 
                                 n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE))
        
        # 2. Convert to DB - Essential for peak finding consistency
        # This makes the loudest peak 0dB and the rest negative.
        spectrogram_db = librosa.amplitude_to_db(cqt, ref=np.max)
        
        # 3. Peak Finding (using -50dB threshold at DEFAULT_AMP_MIN)
        peaks = self._extract_peaks(spectrogram_db, threshold=DEFAULT_AMP_MIN)
        
        # 4. Hash Triplets
        hashes = self._generate_hashes(peaks)
        
        return hashes, time() - t

    def _extract_peaks(self, spectrogram, threshold=DEFAULT_AMP_MIN) -> List[Tuple[int, int]]:
        local_max = maximum_filter(spectrogram, size=(10, 10)) == spectrogram
        background = (spectrogram == 0)
        eroded_background = binary_erosion(background, structure=np.ones((10, 10)))
        detected_peaks = local_max ^ eroded_background
        
        freqs, times = np.where(detected_peaks)
        amps = spectrogram[detected_peaks]
        
        peak_list = []
        for f, t, a in zip(freqs, times, amps):
            if a > threshold:
                peak_list.append((f, t))
        return sorted(peak_list, key=lambda x: x[1])

    def _generate_hashes(self, peaks, fan_out=DEFAULT_FAN_VALUE) -> List[Tuple[str, int]]:
        peaks = np.array(peaks)
        n = len(peaks)
        if n < 3:
            return []

        f = peaks[:, 0].astype(np.int32)
        t = peaks[:, 1]
        
        all_hashes = []

        for j in range(1, fan_out + 1):
            for k in range(j + 1, fan_out + 2):
                idx_limit = n - k
                
                f1, f2, f3 = f[:idx_limit], f[j:j+idx_limit], f[k:k+idx_limit]
                t1, t2, t3 = t[:idx_limit], t[j:j+idx_limit], t[k:k+idx_limit]

                dt21 = t2 - t1
                dt31 = t3 - t1
                df21 = f2 - f1
                df31 = f3 - f1

                valid = dt31 > 0
                if not np.any(valid):
                    continue

                # 1. Calculate the Ratio (Quantized to 3 decimals)
                # We multiply by 1000 to keep 3 decimal places of precision as an int
                t_ratio = (dt21[valid] / dt31[valid] * 1000).astype(np.int32)
                
                v_df21 = df21[valid]
                v_df31 = df31[valid]
                v_t1 = t1[valid].astype(np.int32)

                # 2. Bit-pack into an integer
                packed_hashes = (t_ratio << 16) | ((v_df21 + 100) << 8) | (v_df31 + 100)

                all_hashes.extend(zip(packed_hashes.tolist(), v_t1.tolist()))

        return all_hashes

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
        """
        Optimized for Pitch/Tempo changes. Uses histogram binning to find 
        the strongest alignment peak for each song.
        """
        if not matches:
            return []

        # 1. Group by song_id and apply Histogram Binning
        # matches structure: (song_id, db_offset, query_offset)
        # diff = db_offset - query_offset
        song_diffs = defaultdict(list)
        for sid, db_off, q_off in matches:
            song_diffs[sid].append(db_off - q_off)

        # 2. Find the strongest peak per song
        # bucket_size=5 allows for ~58ms of drift (at 512 hop, 44100 Hz)
        bucket_size = BUCKET_SIZE
        songs_matches = []

        for song_id, diffs in song_diffs.items():
            buckets = defaultdict(int)
            for d in diffs:
                buckets[d // bucket_size] += 1
            
            # Find the best bucket and its count
            best_bucket = max(buckets, key=buckets.get)
            hashes_matched_count = buckets[best_bucket]
            # Use the average or representative offset from that bucket
            representative_offset = best_bucket * bucket_size
            
            songs_matches.append((song_id, representative_offset, hashes_matched_count))

        # 3. Sort by the alignment strength (count)
        songs_matches.sort(key=lambda x: x[2], reverse=True)

        # 4. Build JSON output structure
        songs_result = []
        for song_id, offset, hashes_matched_count in songs_matches[0:topn]:
            song = self.db.get_song_by_id(song_id)
            if not song: continue

            # Handle SHA1 logic (maintaining consistency with your previous fix)
            raw_sha1 = song.get(FIELD_BLOB_SHA1)
            if isinstance(raw_sha1, bytes):
                blob_sha1_str = raw_sha1.hex()
            else:
                blob_sha1_str = str(raw_sha1)

            # Time Calculation: (frame_index * hop_length) / sample_rate
            # Assuming 512 hop and 44100 Hz
            nseconds = round(float(offset) * 512 / 44100, 5)

            total_hashes_in_db = song.get(FIELD_TOTAL_HASHES) or 1

            songs_result.append({
                SONG_ID: str(song_id),
                SONG_NAME: song.get(SONG_NAME),
                INPUT_HASHES: queried_hashes,
                FINGERPRINTED_HASHES: total_hashes_in_db,
                HASHES_MATCHED: hashes_matched_count,
                INPUT_CONFIDENCE: hashes_matched_count / queried_hashes,
                FINGERPRINTED_CONFIDENCE: hashes_matched_count / total_hashes_in_db,
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
