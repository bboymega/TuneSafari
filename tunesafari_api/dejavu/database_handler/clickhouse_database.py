import queue
import abc
import sys
from clickhouse_driver import Client
from typing import Dict, List, Tuple
from dejavu.base_classes.base_database import BaseDatabase
import numpy as np
import traceback
import redis
import json
import pickle
import uuid
import random
from datetime import datetime
from dejavu.config.settings import (FIELD_BLOB_SHA1, FIELD_FINGERPRINTED,
                                    FIELD_HASH, FIELD_OFFSET, FIELD_SONG_ID,
                                    FIELD_SONGNAME, FIELD_TOTAL_HASHES,
                                    FINGERPRINTS_TABLENAME, SONGS_TABLENAME, CONFIG_FILE)

config_file = CONFIG_FILE if CONFIG_FILE not in [None, '', 'config.json'] else 'config.json'

class Query(BaseDatabase, metaclass=abc.ABCMeta):
    def __init__(self, redis_db_index):
        super().__init__()
        self.redis_db_index = redis_db_index
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)

            redis_conf = config_data.get("redis", {})
            host = redis_conf.get("host", "127.0.0.1")
            user = redis_conf.get("user")
            password = redis_conf.get("password")
            port = redis_conf.get("port", 6379)
            self.prefix = redis_conf.get("prefix", "TuneScout")
            db_index = self.redis_db_index

            redis_kwargs = {
                "host": host,
                "port": port,
                "db": db_index,
                "socket_timeout": 2.0,
                "socket_connect_timeout": 2.0,
                "retry_on_timeout": True
            }
            if user:
                redis_kwargs["username"] = user
            if password:
                redis_kwargs["password"] = password

            self.redis_pool = redis.ConnectionPool(**redis_kwargs)
            self.redis_client = redis.Redis(connection_pool=self.redis_pool)
            self.redis_client.ping()
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
            sys.stderr.write(f"\033[33m{datetime.now().strftime("[%d/%b/%Y %H:%M:%S]")} TuneScout \"WARNING: Redis Connection Warning: {e}. Falling back to SQL only mode for recognition\"\033[0m\n")
            self.redis_client = None
    
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
        Uses the INSERT-as-Update pattern for ReplacingMergeTree.
        """
        GET_SONG = f"""
            SELECT 
                {FIELD_SONGNAME}, 
                {FIELD_BLOB_SHA1}, 
                {FIELD_TOTAL_HASHES} 
            FROM `{SONGS_TABLENAME}` FINAL 
            WHERE `{FIELD_SONG_ID}` = %(sid)s
        """
        sid = uuid.UUID(song_id) if isinstance(song_id, str) else song_id
        result = self.client.execute(GET_SONG, {'sid': sid})
        if not result:
            return  # Song not found
        song_name, blob_sha1, total_hashes = result[0]

        UPSERT_QUERY = f"""
            INSERT INTO `{SONGS_TABLENAME}` 
            (`{FIELD_SONG_ID}`, `{FIELD_SONGNAME}`, `{FIELD_FINGERPRINTED}`, 
            `{FIELD_BLOB_SHA1}`, `{FIELD_TOTAL_HASHES}`, `date_modified`) 
            VALUES
        """
        data = [(
            sid, 
            song_name, 
            1,               # The update: fingerprinted = 1
            blob_sha1, 
            total_hashes, 
            datetime.now()
        )]

        self.client.execute(UPSERT_QUERY, data)
        
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
            FROM `{SONGS_TABLENAME}` FINAL
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

    def insert(self, fingerprint: int, song_id, offset: int):
        """
        Inserts a single fingerprint into the database.

        :param fingerprint: Part of a sha1 hash, in hexadecimal format.
        :param song_id: Song identifier this fingerprint is associated with.
        :param offset: The offset this fingerprint is from.
        """
        INSERT_FINGERPRINT = f"""
            INSERT INTO `{FINGERPRINTS_TABLENAME}` (
            `{FIELD_HASH}`,
            `{FIELD_SONG_ID}`,
            `{FIELD_OFFSET}`
            )
            VALUES
        """
        self.client.execute(INSERT_FINGERPRINT, [(fingerprint, song_id, offset)])

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
    
    def query(self, fingerprint: int = None) -> List[Tuple]:
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
    
    def insert_hashes(self, song_id, hashes: List[Tuple[int, int]], batch_size: int = 1000) -> None:
        """
        Insert a multitude of fingerprints.

        :param song_id: Song identifier the fingerprints belong to
        :param hashes: A sequence of tuples in the format (hash, offset)
            - hash: Part of a sha1 hash, in hexadecimal format
            - offset: Offset this hash was created from/at.
        :param batch_size: insert batches.
        """
        # Prepare values to insert
        values = [(hsh, song_id, int(offset)) for hsh, offset in hashes]

        # Batch insert the values
        for index in range(0, len(values), batch_size):
            INSERT_FINGERPRINT = f"""
                INSERT INTO `{FINGERPRINTS_TABLENAME}` (
                `{FIELD_HASH}`,
                `{FIELD_SONG_ID}`,
                `{FIELD_OFFSET}`
                )
                VALUES
            """
            self.client.execute(INSERT_FINGERPRINT, values[index: index + batch_size])
    
    def return_matches(self, hashes: List[Tuple[int, int]], batch_size: int = 1000) -> Tuple[List[Tuple[int, int]], Dict[int, int]]:
        """
        Searches Redis (cache) then ClickHouse (fallback). 
        Uses NumPy for result expansion and fixes the NoneType dedup_hashes error.
        """
        # Prepare Data for Query (Original Logic)
        mapper = {}
        for hsh, offset in hashes:
            if hsh not in mapper:
                mapper[hsh] = []
            mapper[hsh].append(offset)
        
        for hsh in mapper:
            mapper[hsh] = np.array(mapper[hsh], dtype=np.int64)

        values = list(mapper.keys())
        all_sids_flat = []
        all_offsets_diff_flat = []
        dedup_hashes: Dict[int, int] = {}

        for index in range(0, len(values), batch_size):
            current_batch = values[index: index + batch_size]
            
            # Check Redis Cache
            hit_blocks = []
            cache_misses = []
            if self.redis_client:
                pipe = self.redis_client.pipeline()
                for hsh in current_batch:
                    pipe.get(f"{self.prefix}:{hsh}")
                redis_responses = pipe.execute()

                for hsh, raw_data in zip(current_batch, redis_responses):
                    if raw_data:
                        m_list = pickle.loads(raw_data)
                        if m_list:
                            # Convert the first element of each row (the song_id) to a UUID object
                            # This makes it look EXACTLY like the SQL results
                            m_transformed = [[uuid.UUID(str(row[0])), row[1]] for row in m_list]
                            
                            m = np.array(m_transformed, dtype=object) 
                            h_col = np.full((m.shape[0], 1), hsh)
                            hit_blocks.append(np.hstack((h_col, m)))
                    else:
                        cache_misses.append(hsh)
            else:
                for hsh in current_batch:
                    cache_misses.append(hsh)


            sql_block = np.empty((0, 3), dtype=object)
            if cache_misses:
                clean_hashes = [int(h) for h in cache_misses]
                SELECT_MULTIPLE = f"""
                    SELECT `{FIELD_HASH}`, `{FIELD_SONG_ID}`, `{FIELD_OFFSET}`
                    FROM `{FINGERPRINTS_TABLENAME}`
                    WHERE `{FIELD_HASH}` IN %(hashes)s
                """
                sql_results = self.client.execute(SELECT_MULTIPLE, {'hashes': clean_hashes})
                
                if sql_results:
                    sql_block = np.array(sql_results, dtype=object)
                    
                    # POPULATE CACHE (Vectorized Grouping for Redis)
                    sort_idx = np.argsort(sql_block[:, 0])
                    sorted_arr = sql_block[sort_idx]
                    unq_h, indices = np.unique(sorted_arr[:, 0], return_index=True)
                    groups = np.split(sorted_arr, indices[1:])
                    if self.redis_client:
                        write_pipe = self.redis_client.pipeline()
                        for h_key, group in zip(unq_h, groups):
                            # Cache only [sid, offset]
                            redis_key = f"{self.prefix}:{h_key}"
                            write_pipe.setex(redis_key, 86400, pickle.dumps(group[:, [1, 2]].tolist()))
                        write_pipe.execute()

            # Merge Cache & DB results
            all_blocks = [sql_block] + hit_blocks if hit_blocks else [sql_block]
            combined = np.vstack(all_blocks) if any(b.size > 0 for b in all_blocks) else np.empty((0, 3))

            if combined.size == 0:
                continue

            db_hashes = combined[:, 0]
            db_sids = np.array([uuid.UUID(str(s)) for s in combined[:, 1]], dtype=object)
            db_offsets = combined[:, 2].astype(np.int64)

            # Add Missed Hashes to Cache
            u_sids, counts = np.unique(db_sids, return_counts=True)
            for sid, count in zip(u_sids, counts):
                dedup_hashes[sid] = dedup_hashes.get(sid, 0) + count

            unique_hashes_in_batch = np.unique(db_hashes)
            for hsh in unique_hashes_in_batch:
                match_indices = np.where(db_hashes == hsh)[0]
                db_offsets_for_hsh = db_offsets[match_indices]
                db_sids_for_hsh = db_sids[match_indices]
                sampled_offsets = mapper[hsh]

                # Broadcast differences
                diff_matrix = db_offsets_for_hsh[:, None] - sampled_offsets[None, :]
                sid_matrix = np.repeat(db_sids_for_hsh, sampled_offsets.shape[0])
                
                all_sids_flat.extend(sid_matrix)
                all_offsets_diff_flat.extend(diff_matrix.flatten())

        results = list(zip(all_sids_flat, all_offsets_diff_flat)) if all_sids_flat else []
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
            `{FIELD_SONG_ID}` UUID DEFAULT generateUUIDv4(),
            `{FIELD_SONGNAME}` LowCardinality(String),
            `{FIELD_FINGERPRINTED}` UInt8 DEFAULT 0,
            `{FIELD_BLOB_SHA1}` FixedString(40),
            `{FIELD_TOTAL_HASHES}` UInt32 DEFAULT 0 CODEC(T64, LZ4),
            `date_created` DateTime64(3) DEFAULT now64() CODEC(DoubleDelta, LZ4),
            `date_modified` DateTime64(3) DEFAULT now64() CODEC(DoubleDelta, LZ4)
        ) 
        ENGINE = ReplacingMergeTree(date_modified)
        ORDER BY `{FIELD_SONG_ID}`
        SETTINGS index_granularity = 8192;
        """
    
    CREATE_FINGERPRINTS_TABLE = f"""
        CREATE TABLE IF NOT EXISTS `{FINGERPRINTS_TABLENAME}` (
            `{FIELD_HASH}` UInt64 CODEC(Delta, LZ4),
            `{FIELD_SONG_ID}` UUID,
            `{FIELD_OFFSET}` UInt32 CODEC(DoubleDelta, LZ4),
            `date_created` DateTime64(3) DEFAULT now64() CODEC(DoubleDelta, LZ4)
        ) 
        ENGINE = MergeTree()
        ORDER BY (`{FIELD_HASH}`, `{FIELD_SONG_ID}`)
        PARTITION BY tuple()
        SETTINGS index_granularity = 8192;
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
        FROM `{SONGS_TABLENAME}` FINAL
        WHERE `{FIELD_FINGERPRINTED}` = 1;
    """

    SELECT_SONGS = f"""
        SELECT
        `{FIELD_SONG_ID}`,
        `{FIELD_SONGNAME}`,
        `{FIELD_BLOB_SHA1}` AS `{FIELD_BLOB_SHA1}`,
        `{FIELD_TOTAL_HASHES}`,
        `date_created`
        FROM `{SONGS_TABLENAME}` FINAL
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
        redis_db_index = options.pop("redis_db_index", random.randint(0, 15))
        super().__init__(redis_db_index=redis_db_index)
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
                FROM `{SONGS_TABLENAME}` FINAL
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

    