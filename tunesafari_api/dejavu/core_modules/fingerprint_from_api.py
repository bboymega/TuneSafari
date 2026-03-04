import json
import multiprocessing as mp
import hashlib
import sys
from dejavu import Dejavu
from dejavu.database_handler.result_storage import create_mysql_connection, create_clickhouse_connection
from dejavu.database_handler.select_database import select_database
from dejavu.config.settings import (CONFIG_FILE, SONGS_TABLENAME, FIELD_BLOB_SHA1)

config_file = CONFIG_FILE if CONFIG_FILE not in [None, '', 'config.json'] else 'config.json'

def is_fingerprinted(blob, db_config, result_queue):
    if db_config["database_type"] == "clickhouse":
        clickhouse_db_connection = create_clickhouse_connection(db_config["database"])
        if clickhouse_db_connection:
            try:
                blob_hash = hashlib.sha1(blob).hexdigest()
                is_fingerprinted_query = f"""
                SELECT `{FIELD_BLOB_SHA1}` FROM `{SONGS_TABLENAME}`
                WHERE `{FIELD_BLOB_SHA1}` = '{blob_hash}';
                """
                result = clickhouse_db_connection.execute(is_fingerprinted_query)
                if result: # If the submitted track is already fingerprinted, return its hash value
                    result_queue.put(blob_hash)
            except Exception as e:
                sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")

    elif db_config["database_type"] == "mysql":
        mysql_db_connection = create_mysql_connection(db_config["database"])
        if mysql_db_connection:
            try:
                is_fingerprinted_query = f"""
                SELECT `{FIELD_BLOB_SHA1}` FROM `{SONGS_TABLENAME}`
                WHERE `{FIELD_BLOB_SHA1}` = UNHEX(%s);
                """
                blob_hash = hashlib.sha1(blob).hexdigest()
                cursor = mysql_db_connection.cursor()
                cursor.execute(is_fingerprinted_query, (blob_hash,))
                result = cursor.fetchall()
                if result: # If the submitted track is already fingerprinted, return its hash value
                    result_queue.put(blob_hash)
            except Exception as e:
                sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")
                return None

def is_fingerprinted_all(blob):
    with open(config_file, 'r') as jsonFile:
        db_configs = json.load(jsonFile)["instances"]

        processes = []
        result_queue = mp.Queue()

        # Start a new process for each instance
        for item in db_configs:
            process = mp.Process(target=is_fingerprinted, args=(blob, item, result_queue,))
            processes.append(process)
            process.start()

        # Wait for all process to complete
        for process in processes:
            process.join()

        # Collect results
        while not result_queue.empty():
            return result_queue.get()
        
        return None


def fingerprint(blob, song_name, remote_addr):
    with open(config_file, 'r') as json_file:
        is_fingerprinted = is_fingerprinted_all(blob)
        if is_fingerprinted_all(blob):
            return 1, is_fingerprinted # Status code 1: already fingerprinted
        
        db_configs = json.load(json_file)["instances"]
        selected_db = select_database(db_configs, SONGS_TABLENAME) # select the database config with the least record
        instance = Dejavu(selected_db)
        result, file_hash = instance.fingerprint_blob(blob, song_name, remote_addr)
        return result, file_hash
