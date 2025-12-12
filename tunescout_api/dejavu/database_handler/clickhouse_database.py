import queue
import abc
import sys
from clickhouse_driver import Client
from typing import Dict, List, Tuple
from dejavu.base_classes.base_database import BaseDatabase
import numpy as np
import traceback
from dejavu.config.settings import (FIELD_BLOB_SHA1, FIELD_FINGERPRINTED,
                                    FIELD_HASH, FIELD_OFFSET, FIELD_SONG_ID,
                                    FIELD_SONGNAME, FIELD_TOTAL_HASHES,
                                    FINGERPRINTS_TABLENAME, SONGS_TABLENAME)

class Query(BaseDatabase, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
    
    def before_fork(self) -> None:
        """
        Called before the database instance is given to the new process
        """
        pass
    
    def after_fork(self) -> None:
        """
        Called after the database instance has been given to the new process

        This will be called in the new process.
        """
        pass
    
    def setup(self) -> None:
        """
        Called on creation or shortly afterwards.
        """
        try:
            self.client.execute(self.CREATE_SONGS_TABLE)
            self.client.execute(self.CREATE_FINGERPRINTS_TABLE)
            self.client.execute(self.DELETE_UNFINGERPRINTED)
        except Exception as e:
            traceback_info = traceback.format_exc()
            sys.stderr.write("\033[31m" + "\n--- Full Traceback ---" + "\033[0m\n")
            sys.stderr.write("\033[31m" + traceback_info + "\033[0m\n")
            sys.stderr.write("\033[31m----------------------\033[0m\n")
            sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")
    
    def empty(self) -> None:
        """
        Called when the database should be cleared of all data.
        """
        try:
            self.client.execute(self.DROP_FINGERPRINTS)
            self.client.execute(self.DROP_SONGS)

            self.setup()
        except Exception as e:
            traceback_info = traceback.format_exc()
            sys.stderr.write("\033[31m" + "\n--- Full Traceback ---" + "\033[0m\n")
            sys.stderr.write("\033[31m" + traceback_info + "\033[0m\n")
            sys.stderr.write("\033[31m----------------------\033[0m\n")
            sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")

    def delete_unfingerprinted_songs(self) -> None:
        """
        Called to remove any song entries that do not have any fingerprints
        associated with them.
        """
        self.client.execute(self.DELETE_UNFINGERPRINTED)
    
    def get_num_songs(self) -> int:
        """
        Returns the song's count stored.

        :return: the amount of songs in the database.
        """
        self.client.execute(self.SELECT_UNIQUE_SONG_IDS)
        result = cur.fetchone()
        count = result[0] if result else 0

        return count
    
    def get_num_fingerprints(self) -> int:
        """
        Returns the fingerprints' count stored.

        :return: the number of fingerprints in the database.
        """
        self.client.execute(self.SELECT_NUM_FINGERPRINTS)
        result = cur.fetchone()
        count = result[0] if result else 0

        return count
    

    def set_song_fingerprinted(self, song_id):
        """
        Sets a specific song as having all fingerprints in the database.

        :param song_id: song identifier.
        """
        UPDATE_SONG_FINGERPRINTED = f"""
            UPDATE `{SONGS_TABLENAME}` SET `{FIELD_FINGERPRINTED}` = 1 WHERE `{FIELD_SONG_ID}` = '{song_id}';
        """
        self.client.execute(UPDATE_SONG_FINGERPRINTED)
        
    def get_songs(self) -> List[Dict[str, str]]:
        """
        Returns all fully fingerprinted songs in the database

        :return: a dictionary with the songs info.
        """
        return self.client.execute(self.SELECT_SONGS, dictionary=True)
    
    def get_song_by_id(self, song_id) -> Dict[str, str]:
        """
        Brings the song info from the database.

        :param song_id: song identifier.
        :return: a song by its identifier. Result is a dictionary.
        """
        # Execute the query to fetch the song by its ID, returning the result as a dictionary
        SELECT_SONG = f"""
            SELECT
            `{FIELD_SONGNAME}`,
            `{FIELD_BLOB_SHA1}` AS `{FIELD_BLOB_SHA1}`,
            `{FIELD_TOTAL_HASHES}`
            FROM `{SONGS_TABLENAME}`
            WHERE `{FIELD_SONG_ID}` = '{song_id}';
        """
        result = self.client.execute(SELECT_SONG)
        result_dict = {}
        if result:
            result_dict[FIELD_SONGNAME] = result[0][0]
            result_dict[FIELD_BLOB_SHA1] = result[0][1]
            result_dict[FIELD_TOTAL_HASHES] = result[0][2]
        
        # Since we only want one song, return the first row (if available), else None
        return result_dict if result_dict else {}

    def insert(self, fingerprint: str, song_id, offset: int):
        """
        Inserts a single fingerprint into the database.

        :param fingerprint: Part of a sha1 hash, in hexadecimal format.
        :param song_id: Song identifier this fingerprint is associated with.
        :param offset: The offset this fingerprint is from.
        """
        INSERT_FINGERPRINT = f"""
            INSERT INTO `{FINGERPRINTS_TABLENAME}` (
            `{FIELD_SONG_ID}`,
            `{FIELD_HASH}`,
            `{FIELD_OFFSET}`
            )
            VALUES
        """
        self.client.execute(INSERT_FINGERPRINT, [(song_id, fingerprint, offset)])

    @abc.abstractmethod
    def insert_song(self, song_name: str, file_hash: str, total_hashes: int) -> int:
        """
        Inserts a song name into the database, returns the new
        identifier of the song.

        :param song_name: The name of the song.
        :param file_hash: Hash from the fingerprinted file.
        :param total_hashes: amount of hashes to be inserted on fingerprint table.
        :return: the inserted id.
        """
        pass
    
    def query(self, fingerprint: str = None) -> List[Tuple]:
        """
        Returns all matching fingerprint entries associated with
        the given hash as parameter, if None is passed it returns all entries.

        :param fingerprint: part of a sha1 hash, in hexadecimal format
        :return: a list of fingerprint records stored in the db.
        """
        # Execute the query and fetch results directly from the client
        if fingerprint:
            # If a fingerprint is provided, execute the SELECT query with the parameter
            result = self.client.execute(self.SELECT, ({'data': fingerprint},))
        else:
            # If no fingerprint is provided, execute the SELECT_ALL query to fetch all entries
            result = self.client.execute(self.SELECT_ALL)
        
        return result
    
    def get_iterable_kv_pairs(self) -> List[Tuple]:
        """
        Returns all fingerprints in the database.

        :return: a list containing all fingerprints stored in the db.
        """
        return self.query(None)
    
    def insert_hashes(self, song_id, hashes: List[Tuple[str, int]], batch_size: int = 1000) -> None:
        """
        Insert a multitude of fingerprints.

        :param song_id: Song identifier the fingerprints belong to
        :param hashes: A sequence of tuples in the format (hash, offset)
            - hash: Part of a sha1 hash, in hexadecimal format
            - offset: Offset this hash was created from/at.
        :param batch_size: insert batches.
        """
        # Prepare values to insert
        values = [(song_id, hsh, int(offset)) for hsh, offset in hashes]

        # Batch insert the values
        for index in range(0, len(values), batch_size):
            INSERT_FINGERPRINT = f"""
                INSERT INTO `{FINGERPRINTS_TABLENAME}` (
                `{FIELD_SONG_ID}`,
                `{FIELD_HASH}`,
                `{FIELD_OFFSET}`
                )
                VALUES
            """
            self.client.execute(INSERT_FINGERPRINT, values[index: index + batch_size])
    
    def return_matches(self, hashes: List[Tuple[str, int]], batch_size: int = 1000) -> Tuple[List[Tuple[int, int]], Dict[int, int]]:
        """
        Searches the database for pairs of (hash, offset) values, refactored with NumPy.
        """
        # Prepare Data for Query
        mapper = {}
        for hsh, offset in hashes:
            # We convert offsets to NumPy arrays immediately for later vectorization
            if hsh not in mapper:
                mapper[hsh] = []
            mapper[hsh].append(offset)
        
        # Convert lists of offsets into NumPy arrays *once*
        for hsh in mapper:
             mapper[hsh] = np.array(mapper[hsh], dtype=np.int64)

        values = list(mapper.keys())  # All the unique hashes

        dedup_hashes: Dict[int, int] = {}
        results: List[Tuple[int, int]] = [] # Still a Python list, efficient for many appends

        # Batch Query and Vectorized Processing
        for index in range(0, len(values), batch_size):
            current_batch = values[index: index + batch_size]

            SELECT_MULTIPLE = f"""
                SELECT `{FIELD_HASH}`,
                    `{FIELD_SONG_ID}`,
                    `{FIELD_OFFSET}`
                FROM `{FINGERPRINTS_TABLENAME}`
                WHERE `{FIELD_HASH}` IN ({', '.join([repr(hash_value.lower()) for hash_value in current_batch])});
            """

            result = self.client.execute(SELECT_MULTIPLE)
            
            # The database result is a list of (hsh, sid, offset) tuples.
            # We convert it into separate, aligned NumPy arrays for fast processing.
            if not result:
                continue

            # Separate columns into arrays
            db_hashes = np.array([row[0] for row in result])
            db_sids = np.array([row[1] for row in result], dtype=object)
            db_offsets = np.array([row[2] for row in result], dtype=np.int64)

            unique_hashes = np.unique(db_hashes)
            
            for hsh in unique_hashes:
                match_indices = np.where(db_hashes == hsh)[0]
                sampled_offsets = mapper[hsh] # np.array of sampled offsets

                for i in match_indices:
                    sid = db_sids[i]
                    db_offset = db_offsets[i]
                    offset_differences = db_offset - sampled_offsets 
                    
                    sid_array = np.full_like(offset_differences, sid, dtype=object)
                    temp_results = np.column_stack((sid_array, offset_differences))
                    
                    results.extend(map(tuple, temp_results))

                    dedup_hashes[sid] = dedup_hashes.get(sid, 0) + 1 # This counts duplicates based on sid and hash match (if a hash appears twice in the db for the same song, it's counted twice in the original). Assuming this is the desired behavior.

        return results, dedup_hashes
    
    def delete_songs_by_id(self, song_ids, batch_size: int = 1000) -> None:
        """
        Given a list of song ids, it deletes all songs specified and their corresponding fingerprints.

        :param song_ids: song ids to be deleted from the database.
        :param batch_size: number of query batches.
        """
        # Process the song ids in batches
        for index in range(0, len(song_ids), batch_size):
            # Get the current batch of song_ids
            batch = song_ids[index: index + batch_size]
            
            # Construct the delete query (no need to manually join placeholders)
            query = f"""
            DELETE FROM `{FINGERPRINTS_TABLENAME}`
            WHERE `{FIELD_SONG_ID}` IN ('{batch}');
            """
            
            # Execute the query with the batch of song IDs as parameters
            self.client.execute(query)


class ClickhouseDatabase(Query):
    type = "clickhouse"

    # CREATES
    CREATE_SONGS_TABLE = f"""
        CREATE TABLE IF NOT EXISTS `{SONGS_TABLENAME}` (
        `{FIELD_SONG_ID}` UUID NOT NULL DEFAULT generateUUIDv4(),
        `{FIELD_SONGNAME}` String NOT NULL,
        `{FIELD_FINGERPRINTED}` Int8 DEFAULT 0,
        `{FIELD_BLOB_SHA1}` FixedString(40) NOT NULL,
        `{FIELD_TOTAL_HASHES}` Int32 NOT NULL DEFAULT 0,
        `date_created` DateTime DEFAULT now(),
        `date_modified` DateTime DEFAULT now(),
        ) 
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(date_created)
        ORDER BY `{FIELD_SONG_ID}`
        SETTINGS enable_block_number_column = 1, enable_block_offset_column = 1;
    """
    
    CREATE_FINGERPRINTS_TABLE = f"""
        CREATE TABLE IF NOT EXISTS `{FINGERPRINTS_TABLENAME}` (
        `{FIELD_HASH}` FixedString(25) NOT NULL,
        `{FIELD_SONG_ID}` UUID NOT NULL,
        `{FIELD_OFFSET}` UInt32 NOT NULL,
        `date_created` DateTime DEFAULT now(),
        `date_modified` DateTime DEFAULT now()
        ) 
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(date_created)
        ORDER BY (`{FIELD_SONG_ID}`, `{FIELD_OFFSET}`, `{FIELD_HASH}`)
        SETTINGS enable_block_number_column = 1, enable_block_offset_column = 1;
    """
    

    # SELECTS
    SELECT = f"""
        SELECT
        `{FIELD_HASH}`,
        `{FIELD_SONG_ID}`,
        `{FIELD_OFFSET}`
        FROM `{FINGERPRINTS_TABLENAME}`
        WHERE `{FIELD_HASH}` IN (%(data)s);
    """
    

    SELECT_ALL = f"""
        SELECT
        `{FIELD_SONG_ID}`,
        `{FIELD_OFFSET}`
        FROM `{FINGERPRINTS_TABLENAME}`;
    """


    SELECT_NUM_FINGERPRINTS = f"""
        SELECT COUNT(*) AS n FROM `{FINGERPRINTS_TABLENAME}`;
    """

    SELECT_UNIQUE_SONG_IDS = f"""
        SELECT COUNT(`{FIELD_SONG_ID}`) AS n
        FROM `{SONGS_TABLENAME}`
        WHERE `{FIELD_FINGERPRINTED}` = 1;
    """

    SELECT_SONGS = f"""
        SELECT
        `{FIELD_SONG_ID}`,
        `{FIELD_SONGNAME}`,
        `{FIELD_BLOB_SHA1}` AS `{FIELD_BLOB_SHA1}`,
        `{FIELD_TOTAL_HASHES}`,
        `date_created`
        FROM `{SONGS_TABLENAME}`
        WHERE `{FIELD_FINGERPRINTED}` = 1;
    """

    # DROPS
    DROP_FINGERPRINTS = f"""
        DROP TABLE IF EXISTS `{FINGERPRINTS_TABLENAME}`;
    """


    # DELETES
    DELETE_UNFINGERPRINTED = f"""
        DELETE FROM `{SONGS_TABLENAME}` WHERE `{FIELD_FINGERPRINTED}` = 0;
    """

    def __init__(self, **options):
        super().__init__()
        self.client = Client(**options)
        self._options = options
    
    def after_fork(self) -> None:
        Client.clear_cache()

    def insert_song(self, song_name: str, file_hash: str, total_hashes: int) -> int:
        """
        Inserts a song name into the database, returns the new
        identifier of the song.

        :param song_name: The name of the song.
        :param file_hash: Hash from the fingerprinted file.
        :param total_hashes: amount of hashes to be inserted on fingerprint table.
        :return: the inserted id.
        """
        INSERT_SONG = f"""
            INSERT INTO `{SONGS_TABLENAME}` (
            `{FIELD_SONGNAME}`,
            `{FIELD_BLOB_SHA1}`,
            `{FIELD_TOTAL_HASHES}`
            )
            VALUES
        """
        try:
            self.client.execute(INSERT_SONG, [(song_name, file_hash, total_hashes)])

            SELECT_SONG_ID = f"""
                SELECT `{FIELD_SONG_ID}`
                FROM `{SONGS_TABLENAME}`
                WHERE `{FIELD_SONGNAME}` = '{song_name}' AND `{FIELD_BLOB_SHA1}` = '{file_hash}'
                ORDER BY `{FIELD_SONG_ID}` DESC
                LIMIT 1
            """
            result = self.client.execute(SELECT_SONG_ID)

            if result:
                return result[0][0]  # Assuming the result is a list of tuples and the first element is the song_id
        except Exception as e:
            traceback_info = traceback.format_exc()
            sys.stderr.write("\033[31m" + "\n--- Full Traceback ---" + "\033[0m\n")
            sys.stderr.write("\033[31m" + traceback_info + "\033[0m\n")
            sys.stderr.write("\033[31m----------------------\033[0m\n")
            sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")
        
    def __getstate__(self):
        return self._options,

    def __setstate__(self, state):
        self._options, = state
        self.client = Client(**self._options)

    