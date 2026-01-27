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
from scipy.ndimage import maximum_filter, binary_erosion
import dejavu.logic.decoder as decoder
from dejavu.base_classes.base_database import get_database
from dejavu.config.settings import (DEFAULT_FS, DEFAULT_OVERLAP_RATIO,
                                    DEFAULT_WINDOW_SIZE, FIELD_BLOB_SHA1,
                                    FIELD_TOTAL_HASHES,
                                    FINGERPRINTED_CONFIDENCE,
                                    FINGERPRINTED_HASHES, HASHES_MATCHED,
                                    INPUT_CONFIDENCE, INPUT_HASHES, OFFSET,
                                    OFFSET_SECS, SONG_ID, SONG_NAME, TOPN, DEFAULT_FAN_VALUE)
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
        cqt = np.abs(librosa.cqt(samples_float, sr=Fs, hop_length=512, 
                                 fmin=librosa.note_to_hz('C1'), 
                                 n_bins=84, bins_per_octave=12))
        
        # 2. Convert to DB - Essential for peak finding consistency
        # This makes the loudest peak 0dB and the rest negative.
        spectrogram_db = librosa.amplitude_to_db(cqt, ref=np.max)
        
        # 3. Peak Finding (using -50dB threshold)
        peaks = self._extract_peaks(spectrogram_db, threshold=-50)
        
        # 4. Hash Triplets
        hashes = self._generate_hashes(peaks)
        
        return hashes, time() - t

    def _extract_peaks(self, spectrogram, threshold=0.1) -> List[Tuple[int, int]]:
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
        fingerprints = []
        for i in range(len(peaks) - (fan_out + 1)):
            for j in range(1, fan_out + 1):
                for k in range(j + 1, fan_out + 2):
                    p1, p2, p3 = peaks[i], peaks[i+j], peaks[i+k]
                    
                    # Time Ratio (Speed Invariant)
                    dt21 = p2[1] - p1[1]
                    dt31 = p3[1] - p1[1]
                    if dt31 <= 0: continue
                    time_ratio = dt21 / dt31

                    # Frequency Delta (Pitch Invariant)
                    df21 = p2[0] - p1[0]
                    df31 = p3[0] - p1[0]

                    # Create a 20-char SHA1 hash of the triplet geometry
                    h_str = f"{time_ratio:.3f}|{df21}|{df31}"
                    h = hashlib.sha1(h_str.encode()).hexdigest()[:20]
                    fingerprints.append((h, p1[1])) # (hash, offset)
        return fingerprints

    def find_matches(self, hashes: List[Tuple[str, int]]) -> Tuple[List[Tuple[int, int]], Dict[str, int], float]:
        """
        Finds the corresponding matches on the fingerprinted audios for the given hashes.

        :param hashes: list of tuples for hashes and their corresponding offsets
        :return: a tuple containing the matches found against the db, a dictionary which counts the different
         hashes matched for each song (with the song id as key), and the time that the query took.

        """
        t = time()
        matches, dedup_hashes = self.db.return_matches(hashes)
        query_time = time() - t
        return matches, dedup_hashes, query_time

    def align_matches(self, matches: List[Tuple[int, int]], dedup_hashes: Dict[str, int], queried_hashes: int,
                  topn: int = TOPN) -> List[Dict[str, any]]:
        """
        Handles the binary/string SHA1 issue and 
        calculates timing based on CQT hop_length.
        """
        # 1. Group by song_id and time-offset
        sorted_matches = sorted(matches, key=lambda m: (m[0], m[1]))
        counts = [(*key, len(list(group))) for key, group in groupby(sorted_matches, key=lambda m: (m[0], m[1]))]
        
        # 2. Find the best offset (histogram peak) for each song
        songs_matches = sorted(
            [max(list(group), key=lambda g: g[2]) for key, group in groupby(counts, key=lambda count: count[0])],
            key=lambda count: count[2], reverse=True
        )

        songs_result = []
        for song_id, offset, hashes_matched_count in songs_matches[0:topn]:
            song = self.db.get_song_by_id(song_id)
            if not song: continue

            # --- FIX: SHA1 Attribute Error ---
            # Ensure we return the SHA1 in a format the rest of your API expects
            raw_sha1 = song.get(FIELD_BLOB_SHA1)
            if isinstance(raw_sha1, bytes):
                # If your API expects a hex string, keep it as bytes so .hex() works later
                # OR convert it here. To match your error fix, let's keep it consistent:
                blob_sha1 = raw_sha1
            else:
                # If it's already a string, we might need to wrap it or handle it in the API
                blob_sha1 = raw_sha1

            # --- FIX: Time Calculation ---
            # Uses CQT. The 'offset' is the frame index.
            # Time (sec) = (frame_index * hop_length) / sample_rate
            nseconds = round(float(offset) * 512 / DEFAULT_FS, 5)

            songs_result.append({
                SONG_ID: song_id,
                SONG_NAME: song.get(SONG_NAME),
                INPUT_HASHES: queried_hashes,
                FINGERPRINTED_HASHES: song.get(FIELD_TOTAL_HASHES),
                HASHES_MATCHED: hashes_matched_count, # Use the actual count from the histogram
                INPUT_CONFIDENCE: hashes_matched_count / queried_hashes,
                FINGERPRINTED_CONFIDENCE: hashes_matched_count / (song.get(FIELD_TOTAL_HASHES) or 1),
                OFFSET: max(0, int(offset)),
                OFFSET_SECS: max(0, nseconds),
                FIELD_BLOB_SHA1: blob_sha1.lower()
            })

        return songs_result

    def recognize(self, recognizer, *options, **kwoptions) -> Dict[str, any]:
        r = recognizer(self)
        return r.recognize(*options, **kwoptions)

    @staticmethod
    def _fingerprint_worker(blob, song_name, remote_addr):
        # We instantiate a temporary Dejavu object to use the instance methods
        # or simply point to a standalone function. 
        # For this compatibility, let's assume we call a revised get_blob_fingerprints
        channels, fs, file_hash = decoder.read(blob)
        # Assuming you've moved the logic to a place accessible here
        # For brevity, you would call the generate_fingerprints logic here
        return [], file_hash # Placeholder for actual worker logic

    @staticmethod
    def get_blob_fingerprints(blob, song_name, remote_addr, print_output: bool = False):
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
