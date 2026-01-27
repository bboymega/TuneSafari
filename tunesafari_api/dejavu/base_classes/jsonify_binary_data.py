import numpy as np
import uuid
from dejavu.config.settings import FIELD_BLOB_SHA1, FIELD_SONG_ID

def jsonify_binary(data):
    # Handle np.int64 and byte strings
    for key, value in data.items():
        if isinstance(value, np.int64):
            data[key] = int(value)  # Convert np.int64 to int
        elif isinstance(value, uuid.UUID):
            data[key] = str(value)
        elif isinstance(value, bytes): # Convert bytes to string
            if key == FIELD_BLOB_SHA1:
                data[key] = value.decode('utf-8').lower() #sha1 value should be lowercase
            else:
                data[key] = value.decode('utf-8')
    return data
