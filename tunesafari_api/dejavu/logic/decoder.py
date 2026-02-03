import fnmatch
import os, io
from hashlib import sha1
from typing import List, Tuple

import numpy as np
from pydub import AudioSegment
from pydub.utils import audioop

from dejavu.third_party import wavio


def unique_hash(blob, block_size: int = 2**20) -> str:
    s = sha1()
    blob_file = io.BytesIO(blob)
    blob_file.seek(0)
    # Read the binary data in chunks and update the hash
    while True:
        buf = blob_file.read(block_size)
        if not buf:
            break
        s.update(buf)
    return s.hexdigest().upper()

def find_files(path: str, extensions: List[str]) -> List[Tuple[str, str]]:
    """
    Get all files that meet the specified extensions.

    :param path: path to a directory with audio files.
    :param extensions: file extensions to look for.
    :return: a list of tuples with file name and its extension.
    """
    # Allow both with ".mp3" and without "mp3" to be used for extensions
    extensions = [e.replace(".", "") for e in extensions]

    results = []
    for dirpath, dirnames, files in os.walk(path):
        for extension in extensions:
            for f in fnmatch.filter(files, f"*.{extension}"):
                p = os.path.join(dirpath, f)
                results.append((p, extension))
    return results


def read(blob, limit: int = None) -> Tuple[List[List[int]], int, str]:

    # pydub does not support 24-bit wav files, use wavio when this occurs
    try:
        audio_data = io.BytesIO(blob)
        audio_data.seek(0)
        audiofile = AudioSegment.from_file(audio_data)

        if limit:
            audiofile = audiofile[:limit * 1000]

        data = np.frombuffer(audiofile.raw_data, np.int16)

        channels = []
        for chn in range(audiofile.channels):
            channels.append(data[chn::audiofile.channels])

        audiofile.frame_rate
    except audioop.error:
        _, _, audiofile = wavio.readwav(audio_data)

        if limit:
            audiofile = audiofile[:limit * 1000]

        audiofile = audiofile.T
        audiofile = audiofile.astype(np.int16)

        channels = []
        for chn in audiofile:
            channels.append(chn)

    return channels, audiofile.frame_rate, unique_hash(blob)


def get_audio_name_from_path(file_path: str) -> str:
    """
    Extracts song name from a file path.

    :param file_path: path to an audio file.
    :return: file name
    """
    return os.path.splitext(os.path.basename(file_path))[0]
