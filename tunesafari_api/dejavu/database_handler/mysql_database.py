import queue
import abc
import sys
import mysql.connector
import numpy as np
import traceback
import redis
import json
import pickle
import uuid
import random
from datetime import datetime
from typing import Dict, List, Tuple
from dejavu.base_classes.base_database import BaseDatabase
from mysql.connector.errors import DatabaseError
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
            with self.cursor() as cur:
                cur.execute(self.CREATE_SONGS_TABLE)
                cur.execute(self.CREATE_FINGERPRINTS_TABLE)
                cur.execute(self.DELETE_UNFINGERPRINTED)
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
            with self.cursor() as cur:
                cur.execute(self.DROP_FINGERPRINTS)
                cur.execute(self.DROP_SONGS)

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
        with self.cursor() as cur:
            cur.execute(self.DELETE_UNFINGERPRINTED)

    def get_num_songs(self) -> int:
        """
        Returns the song's count stored.

        :return: the amount of songs in the database.
        """
        with self.cursor(buffered=True) as cur:
            cur.execute(self.SELECT_UNIQUE_SONG_IDS)
            count = cur.fetchone()[0] if cur.rowcount != 0 else 0

        return count

    def get_num_fingerprints(self) -> int:
        """
        Returns the fingerprints' count stored.

        :return: the number of fingerprints in the database.
        """
        with self.cursor(buffered=True) as cur:
            cur.execute(self.SELECT_NUM_FINGERPRINTS)
            count = cur.fetchone()[0] if cur.rowcount != 0 else 0

        return count

    def set_song_fingerprinted(self, song_id):
        """
        Sets a specific song as having all fingerprints in the database.

        :param song_id: song identifier.
        """
        with self.cursor() as cur:
            cur.execute(self.UPDATE_SONG_FINGERPRINTED, (song_id,))

    def get_songs(self) -> List[Dict[str, str]]:
        """
        Returns all fully fingerprinted songs in the database

        :return: a dictionary with the songs info.
        """
        with self.cursor(dictionary=True) as cur:
            cur.execute(self.SELECT_SONGS)
            return list(cur)

    def get_song_by_id(self, song_id) -> Dict[str, str]:
        """
        Brings the song info from the database.

        :param song_id: song identifier.
        :return: a song by its identifier. Result must be a Dictionary.
        """
        with self.cursor(dictionary=True) as cur:
            cur.execute(self.SELECT_SONG, (song_id,))
            return cur.fetchone()

    def insert(self, fingerprint, song_id, offset):
        """
        Inserts a single fingerprint into the database.

        :param fingerprint: Part of a sha1 hash, in hexadecimal format
        :param song_id: Song identifier this fingerprint is off
        :param offset: The offset this fingerprint is from.
        """
        with self.cursor() as cur:
            cur.execute(self.INSERT_FINGERPRINT, (int(fingerprint), song_id, int(offset)))

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
        with self.cursor() as cur:
            if fingerprint:
                cur.execute(self.SELECT, (fingerprint,))
            else:  # select all if no key
                cur.execute(self.SELECT_ALL)
            return list(cur)

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
        values = [(int(hsh), song_id, int(offset)) for hsh, offset in hashes]
        with self.cursor() as cur:
            for index in range(0, len(hashes), batch_size):
                cur.executemany(self.INSERT_FINGERPRINT, values[index: index + batch_size])

    def return_matches(self, hashes: List[Tuple[int, int]], batch_size: int = 1000) -> Tuple[List[Tuple[int, int]], Dict[int, int]]:
        """
        Searches Redis (cache) then MySQL (fallback). 
        Uses NumPy for vectorized result expansion and offset broadcasting.
        """
        # 1. Prepare Data for Query (Mapping input hashes to their offsets)
        mapper = {}
        for hsh, offset in hashes:
            h_int = int(hsh) # Ensure hash is an int for mapping
            if h_int not in mapper:
                mapper[h_int] = []
            mapper[h_int].append(offset)
        
        for hsh in mapper:
            mapper[hsh] = np.array(mapper[hsh], dtype=np.int64)

        values = list(mapper.keys())
        all_sids_flat = []
        all_offsets_diff_flat = []
        dedup_hashes: Dict[int, int] = {}

        for index in range(0, len(values), batch_size):
            current_batch = values[index: index + batch_size]
            
            hit_blocks = []
            cache_misses = []
            
            # Check Redis Cache
            if self.redis_client:
                pipe = self.redis_client.pipeline()
                for hsh in current_batch:
                    pipe.get(f"{self.prefix}:{hsh}")
                redis_responses = pipe.execute()

                for hsh, raw_data in zip(current_batch, redis_responses):
                    if raw_data:
                        m_list = pickle.loads(raw_data)
                        if m_list:
                            # FIX: Cast to string immediately to avoid UUID comparison errors
                            m = np.array(m_list, dtype=object)
                            # Ensure SID column (index 0 in m) is strings
                            m[:, 0] = m[:, 0].astype(str)
                            
                            h_col = np.full((m.shape[0], 1), hsh, dtype=object)
                            hit_blocks.append(np.hstack((h_col, m)))
                    else:
                        cache_misses.append(hsh)
            else:
                cache_misses = current_batch

            # Handle Cache Misses with MySQL
            sql_block = np.empty((0, 3), dtype=object)
            if cache_misses:
                with self.cursor() as cur:
                    query = self.SELECT_MULTIPLE % ', '.join([self.IN_MATCH] * len(cache_misses))
                    cur.execute(query, cache_misses)
                    sql_results = cur.fetchall()
                    
                    if sql_results:
                        # sql_results is [(hash, sid, offset), ...]
                        sql_block = np.array(sql_results, dtype=object)
                        # FIX: Convert UUID/Binary SID to string immediately
                        sql_block[:, 1] = sql_block[:, 1].astype(str)
                        
                        # POPULATE REDIS
                        sort_idx = np.argsort(sql_block[:, 0].astype(np.uint64))
                        sorted_arr = sql_block[sort_idx]
                        unq_h, indices = np.unique(sorted_arr[:, 0].astype(np.uint64), return_index=True)
                        groups = np.split(sorted_arr, indices[1:])
                        
                        if self.redis_client:
                            write_pipe = self.redis_client.pipeline()
                            for h_key, group in zip(unq_h, groups):
                                redis_key = f"{self.prefix}:{h_key}"
                                # Cache [sid_str, offset]
                                write_pipe.setex(redis_key, 86400, pickle.dumps(group[:, [1, 2]].tolist()))
                            write_pipe.execute()

            # 4. Merge Cache & DB results
            all_blocks = [sql_block] + hit_blocks if hit_blocks else [sql_block]
            valid_blocks = [b for b in all_blocks if b.size > 0]
            combined = np.vstack(valid_blocks) if valid_blocks else np.empty((0, 3))

            if combined.size == 0:
                continue

            # --- THE FIX: Uniform Type Casting ---
            db_hashes = combined[:, 0].astype(np.uint64)
            db_sids = combined[:, 1].astype(str) # Strings are safe for np.unique
            db_offsets = combined[:, 2].astype(np.int64)

            # 5. Vectorized Dedup Counting
            u_sids, counts = np.unique(db_sids, return_counts=True)
            for sid, count in zip(u_sids, counts):
                dedup_hashes[sid] = dedup_hashes.get(sid, 0) + int(count)

            # 6. NumPy Broadcasting
            unique_hashes_in_batch = np.unique(db_hashes)
            for hsh in unique_hashes_in_batch:
                match_indices = np.where(db_hashes == hsh)[0]
                db_offsets_for_hsh = db_offsets[match_indices]
                db_sids_for_hsh = db_sids[match_indices]
                sampled_offsets = mapper.get(int(hsh))

                if sampled_offsets is not None:
                    diff_matrix = db_offsets_for_hsh[:, None] - sampled_offsets[None, :]
                    sid_matrix = np.repeat(db_sids_for_hsh, sampled_offsets.shape[0])
                    
                    all_sids_flat.extend(sid_matrix)
                    all_offsets_diff_flat.extend(diff_matrix.flatten().tolist())

        results = list(zip(all_sids_flat, all_offsets_diff_flat)) if all_sids_flat else []
        return results, dedup_hashes

    def delete_songs_by_id(self, song_ids, batch_size: int = 1000) -> None:
        """
        Given a list of song ids it deletes all songs specified and their corresponding fingerprints.

        :param song_ids: song ids to be deleted from the database.
        :param batch_size: number of query's batches.
        """
        with self.cursor() as cur:
            for index in range(0, len(song_ids), batch_size):
                # Create our IN part of the query
                query = self.DELETE_SONGS % ', '.join(['%s'] * len(song_ids[index: index + batch_size]))

                cur.execute(query, song_ids[index: index + batch_size])

class MySQLDatabase(Query):
    type = "mysql"

    # CREATES
    CREATE_SONGS_TABLE = f"""
        CREATE TABLE IF NOT EXISTS `{SONGS_TABLENAME}` (
            `{FIELD_SONG_ID}` VARCHAR(36) NOT NULL DEFAULT (UUID()),
            `{FIELD_SONGNAME}` VARCHAR(250) NOT NULL,
            `{FIELD_FINGERPRINTED}` TINYINT UNSIGNED DEFAULT 0,
            `{FIELD_BLOB_SHA1}` BINARY(20) NOT NULL,
            `{FIELD_TOTAL_HASHES}` INT UNSIGNED NOT NULL DEFAULT 0,
            `date_created` DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
            `date_modified` DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
            PRIMARY KEY (`{FIELD_SONG_ID}`),
            INDEX `idx_sha1` (`{FIELD_BLOB_SHA1}`)
        ) ENGINE=INNODB;
        """
    
    CREATE_FINGERPRINTS_TABLE = f"""
        CREATE TABLE IF NOT EXISTS `{FINGERPRINTS_TABLENAME}` (
            `{FIELD_HASH}` BIGINT UNSIGNED NOT NULL,
            `{FIELD_SONG_ID}` VARCHAR(36) NOT NULL,
            `{FIELD_OFFSET}` INT UNSIGNED NOT NULL,
            `date_created` DATETIME(3) DEFAULT CURRENT_TIMESTAMP(3),
            -- Indexing strategy for fast lookups by Hash
            INDEX `idx_hash` (`{FIELD_HASH}`),
            -- Composite index to speed up song deletions/lookups
            INDEX `idx_song_hash` (`{FIELD_SONG_ID}`, `{FIELD_HASH}`),
            CONSTRAINT `fk_song` FOREIGN KEY (`{FIELD_SONG_ID}`) 
                REFERENCES `{SONGS_TABLENAME}` (`{FIELD_SONG_ID}`) ON DELETE CASCADE
        ) ENGINE=InnoDB;
        """

    # INSERTS (IGNORES DUPLICATES)
    INSERT_FINGERPRINT = f"""
        INSERT IGNORE INTO `{FINGERPRINTS_TABLENAME}` (
                `{FIELD_HASH}`
            ,   `{FIELD_SONG_ID}`
            ,   `{FIELD_OFFSET}`)
        VALUES (%s, %s, %s);
    """

    INSERT_SONG = f"""
        INSERT INTO `{SONGS_TABLENAME}` (`{FIELD_SONGNAME}`,`{FIELD_BLOB_SHA1}`,`{FIELD_TOTAL_HASHES}`)
        VALUES (%s, UNHEX(%s), %s);
    """

    # SELECTS
    SELECT = f"""
        SELECT `{FIELD_SONG_ID}`, `{FIELD_OFFSET}`
        FROM `{FINGERPRINTS_TABLENAME}`
        WHERE `{FIELD_HASH}` = %s;
    """

    SELECT_MULTIPLE = f"""
        SELECT `{FIELD_HASH}`, `{FIELD_SONG_ID}`, `{FIELD_OFFSET}`
        FROM `{FINGERPRINTS_TABLENAME}`
        WHERE `{FIELD_HASH}` IN (%s);
    """

    SELECT_ALL = f"SELECT `{FIELD_SONG_ID}`, `{FIELD_OFFSET}` FROM `{FINGERPRINTS_TABLENAME}`;"

    SELECT_SONG = f"""
        SELECT `{FIELD_SONGNAME}`, HEX(`{FIELD_BLOB_SHA1}`) AS `{FIELD_BLOB_SHA1}`, `{FIELD_TOTAL_HASHES}`
        FROM `{SONGS_TABLENAME}`
        WHERE `{FIELD_SONG_ID}` = %s;
    """

    SELECT_NUM_FINGERPRINTS = f"SELECT COUNT(*) AS n FROM `{FINGERPRINTS_TABLENAME}`;"

    SELECT_UNIQUE_SONG_IDS = f"""
        SELECT COUNT(`{FIELD_SONG_ID}`) AS n
        FROM `{SONGS_TABLENAME}`
        WHERE `{FIELD_FINGERPRINTED}` = 1;
    """

    SELECT_SONGS = f"""
        SELECT
            `{FIELD_SONG_ID}`
        ,   `{FIELD_SONGNAME}`
        ,   HEX(`{FIELD_BLOB_SHA1}`) AS `{FIELD_BLOB_SHA1}`
        ,   `{FIELD_TOTAL_HASHES}`
        ,   `date_created`
        FROM `{SONGS_TABLENAME}`
        WHERE `{FIELD_FINGERPRINTED}` = 1;
    """

    # DROPS
    DROP_FINGERPRINTS = f"DROP TABLE IF EXISTS `{FINGERPRINTS_TABLENAME}`;"
    DROP_SONGS = f"DROP TABLE IF EXISTS `{SONGS_TABLENAME}`;"

    # UPDATE
    UPDATE_SONG_FINGERPRINTED = f"""
        UPDATE `{SONGS_TABLENAME}` SET `{FIELD_FINGERPRINTED}` = 1 WHERE `{FIELD_SONG_ID}` = %s;
    """

    # DELETES
    DELETE_UNFINGERPRINTED = f"""
        DELETE FROM `{SONGS_TABLENAME}` WHERE `{FIELD_FINGERPRINTED}` = 0;
    """

    DELETE_SONGS = f"""
        DELETE FROM `{SONGS_TABLENAME}` WHERE `{FIELD_SONG_ID}` IN (%s);
    """

    # IN
    IN_MATCH = f"%s"

    def __init__(self, **options):
        redis_db_index = options.pop("redis_db_index", random.randint(0, 15))
        super().__init__(redis_db_index=redis_db_index)
        self.cursor = cursor_factory(**options)
        self._options = options

    def after_fork(self) -> None:
        # Clear the cursor cache, we don't want any stale connections from
        # the previous process.
        Cursor.clear_cache()

    def insert_song(self, song_name: str, file_hash: str, total_hashes: int) -> int:
        """
        Inserts a song name into the database, returns the new
        identifier of the song.

        :param song_name: The name of the song.
        :param file_hash: Hash from the fingerprinted file.
        :param total_hashes: amount of hashes to be inserted on fingerprint table.
        :return: the inserted id.
        """
        with self.cursor() as cur:
            try:
                cur.execute(self.INSERT_SONG, (song_name, file_hash, total_hashes))
                query = f"""
                    SELECT `{FIELD_SONG_ID}` FROM `{SONGS_TABLENAME}` WHERE `{FIELD_BLOB_SHA1}` = UNHEX(%s);
                """
                cur.execute(query, (file_hash,))
                result = cur.fetchone()
                if result:
                    return result[0]  # This will return the song_id in UUID format
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
        self.cursor = cursor_factory(**self._options)


def cursor_factory(**factory_options):
    def cursor(**options):
        options.update(factory_options)
        return Cursor(**options)
    return cursor


class Cursor(object):
    """
    Establishes a connection to the database and returns an open cursor.
    # Use as context manager
    with Cursor() as cur:
        cur.execute(query)
        ...
    """
    def __init__(self, dictionary=False, **options):
        super().__init__()

        self._cache = queue.Queue(maxsize=5)

        try:
            conn = self._cache.get_nowait()
            # Ping the connection before using it from the cache.
            conn.ping(True)
        except queue.Empty:
            conn = mysql.connector.connect(**options)

        self.conn = conn
        self.dictionary = dictionary

    @classmethod
    def clear_cache(cls):
        cls._cache = queue.Queue(maxsize=5)

    def __enter__(self):
        self.cursor = self.conn.cursor(dictionary=self.dictionary)
        return self.cursor

    def __exit__(self, extype, exvalue, traceback):
        # if we had a MySQL related error we try to rollback the cursor.
        if extype is DatabaseError:
            self.cursor.rollback()

        self.cursor.close()
        self.conn.commit()

        # Put it back on the queue
        try:
            self._cache.put_nowait(self.conn)
        except queue.Full:
            self.conn.close()
            