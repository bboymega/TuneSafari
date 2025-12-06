import queue
import abc
import sys
import mysql.connector
from typing import Dict, List, Tuple
from dejavu.base_classes.base_database import BaseDatabase
from mysql.connector.errors import DatabaseError
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
            with self.cursor() as cur:
                cur.execute(self.CREATE_SONGS_TABLE)
                cur.execute(self.CREATE_FINGERPRINTS_TABLE)
                cur.execute(self.DELETE_UNFINGERPRINTED)
        except Exception as e:
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
            cur.execute(self.INSERT_FINGERPRINT, (song_id, fingerprint, int(offset)))

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

    def insert_hashes(self, song_id, hashes: List[Tuple[str, int]], batch_size: int = 1000) -> None:
        """
        Insert a multitude of fingerprints.

        :param song_id: Song identifier the fingerprints belong to
        :param hashes: A sequence of tuples in the format (hash, offset)
            - hash: Part of a sha1 hash, in hexadecimal format
            - offset: Offset this hash was created from/at.
        :param batch_size: insert batches.
        """
        values = [(song_id, hsh, int(offset)) for hsh, offset in hashes]
        with self.cursor() as cur:
            for index in range(0, len(hashes), batch_size):
                cur.executemany(self.INSERT_FINGERPRINT, values[index: index + batch_size])

    def return_matches(self, hashes: List[Tuple[str, int]],
                       batch_size: int = 1000) -> Tuple[List[Tuple[int, int]], Dict[int, int]]:
        """
        Searches the database for pairs of (hash, offset) values.

        :param hashes: A sequence of tuples in the format (hash, offset)
            - hash: Part of a sha1 hash, in hexadecimal format
            - offset: Offset this hash was created from/at.
        :param batch_size: number of query's batches.
        :return: a list of (sid, offset_difference) tuples and a
        dictionary with the amount of hashes matched (not considering
        duplicated hashes) in each song.
            - song id: Song identifier
            - offset_difference: (database_offset - sampled_offset)
        """
        # Create a dictionary of hash => offset pairs for later lookups
        mapper = {}
        for hsh, offset in hashes:
            hsh_upper = hsh.upper()
            if hsh_upper in mapper.keys():
                mapper[hsh_upper].append(offset)
            else:
                mapper[hsh_upper] = [offset]

        values = list(mapper.keys())

        # in order to count each hash only once per db offset we use the dic below
        dedup_hashes = {}

        results = []
        with self.cursor() as cur:
            for index in range(0, len(values), batch_size):
                # Create our IN part of the query
                query = self.SELECT_MULTIPLE % ', '.join([self.IN_MATCH] * len(values[index: index + batch_size]))

                cur.execute(query, values[index: index + batch_size])

                for hsh, sid, offset in cur:
                    if sid not in dedup_hashes.keys():
                        dedup_hashes[sid] = 1
                    else:
                        dedup_hashes[sid] += 1
                    #  we now evaluate all offset for each  hash matched
                    for song_sampled_offset in mapper[hsh]:
                        results.append((sid, offset - song_sampled_offset))

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
            `{FIELD_SONG_ID}` VARCHAR(36) NOT NULL DEFAULT (UUID())
        ,   `{FIELD_SONGNAME}` VARCHAR(250) NOT NULL
        ,   `{FIELD_FINGERPRINTED}` TINYINT DEFAULT 0
        ,   `{FIELD_BLOB_SHA1}` BINARY(20) NOT NULL
        ,   `{FIELD_TOTAL_HASHES}` INT NOT NULL DEFAULT 0
        ,   `date_created` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        ,   `date_modified` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ,   CONSTRAINT `pk_{SONGS_TABLENAME}_{FIELD_SONG_ID}` PRIMARY KEY (`{FIELD_SONG_ID}`)
        ,   CONSTRAINT `uq_{SONGS_TABLENAME}_{FIELD_SONG_ID}` UNIQUE KEY (`{FIELD_SONG_ID}`)
        ) ENGINE=INNODB;
    """

    CREATE_FINGERPRINTS_TABLE = f"""
        CREATE TABLE IF NOT EXISTS `{FINGERPRINTS_TABLENAME}` (
            `{FIELD_HASH}` BINARY(10) NOT NULL
        ,   `{FIELD_SONG_ID}` VARCHAR(36) NOT NULL
        ,   `{FIELD_OFFSET}` INT UNSIGNED NOT NULL
        ,   `date_created` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        ,   `date_modified` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ,   INDEX `ix_{FINGERPRINTS_TABLENAME}_{FIELD_HASH}` (`{FIELD_HASH}`)
        ,   CONSTRAINT `uq_{FINGERPRINTS_TABLENAME}_{FIELD_SONG_ID}_{FIELD_OFFSET}_{FIELD_HASH}`
                UNIQUE KEY  (`{FIELD_SONG_ID}`, `{FIELD_OFFSET}`, `{FIELD_HASH}`)
        ,   CONSTRAINT `fk_{FINGERPRINTS_TABLENAME}_{FIELD_SONG_ID}` FOREIGN KEY (`{FIELD_SONG_ID}`)
                REFERENCES `{SONGS_TABLENAME}`(`{FIELD_SONG_ID}`) ON DELETE CASCADE
    ) ENGINE=INNODB;
    """

    # INSERTS (IGNORES DUPLICATES)
    INSERT_FINGERPRINT = f"""
        INSERT IGNORE INTO `{FINGERPRINTS_TABLENAME}` (
                `{FIELD_SONG_ID}`
            ,   `{FIELD_HASH}`
            ,   `{FIELD_OFFSET}`)
        VALUES (%s, UNHEX(%s), %s);
    """

    INSERT_SONG = f"""
        INSERT INTO `{SONGS_TABLENAME}` (`{FIELD_SONGNAME}`,`{FIELD_BLOB_SHA1}`,`{FIELD_TOTAL_HASHES}`)
        VALUES (%s, UNHEX(%s), %s);
    """

    # SELECTS
    SELECT = f"""
        SELECT `{FIELD_SONG_ID}`, `{FIELD_OFFSET}`
        FROM `{FINGERPRINTS_TABLENAME}`
        WHERE `{FIELD_HASH}` = UNHEX(%s);
    """

    SELECT_MULTIPLE = f"""
        SELECT HEX(`{FIELD_HASH}`), `{FIELD_SONG_ID}`, `{FIELD_OFFSET}`
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
    IN_MATCH = f"UNHEX(%s)"

    def __init__(self, **options):
        super().__init__()
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
            