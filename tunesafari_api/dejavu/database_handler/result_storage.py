import mysql.connector
from mysql.connector import Error
from clickhouse_driver import Client
import json
import queue
import sys
import multiprocessing as mp
from dejavu.base_classes.jsonify_binary_data import jsonify_binary
from dejavu.database_handler.select_database import select_database
from dejavu.config.settings import (CONFIG_FILE,
                                    FIELD_RESULT_ID, FIELD_RESULT_TOKEN,

                                    FIELD_BLOB_SHA1, FINGERPRINTED_CONFIDENCE, FINGERPRINTED_HASHES,
                                    HASHES_MATCHED, INPUT_CONFIDENCE, INPUT_HASHES,
                                    FIELD_OFFSET, OFFSET_SECS, FIELD_SONG_ID, FIELD_SONGNAME,

                                    FIELD_RESULT1_BLOB_SHA1, FIELD_RESULT1_FINGERPRINTED_CONFIDENCE,
                                    FIELD_RESULT1_FINGERPRINTED_HASHES_IN_DB, FIELD_RESULT1_HASHES_MATCHED_IN_INPUT,
                                    FIELD_RESULT1_INPUT_CONFIDENCE, FIELD_RESULT1_INPUT_TOTAL_HASHES,
                                    FIELD_RESULT1_OFFSET, FIELD_RESULT1_OFFSET_SECONDS,
                                    FIELD_RESULT1_SONG_ID, FIELD_RESULT1_SONG_NAME,

                                    FIELD_RESULT2_BLOB_SHA1, FIELD_RESULT2_FINGERPRINTED_CONFIDENCE,
                                    FIELD_RESULT2_FINGERPRINTED_HASHES_IN_DB, FIELD_RESULT2_HASHES_MATCHED_IN_INPUT,
                                    FIELD_RESULT2_INPUT_CONFIDENCE, FIELD_RESULT2_INPUT_TOTAL_HASHES,
                                    FIELD_RESULT2_OFFSET, FIELD_RESULT2_OFFSET_SECONDS,
                                    FIELD_RESULT2_SONG_ID, FIELD_RESULT2_SONG_NAME,

                                    FIELD_RESULT3_BLOB_SHA1, FIELD_RESULT3_FINGERPRINTED_CONFIDENCE,
                                    FIELD_RESULT3_FINGERPRINTED_HASHES_IN_DB, FIELD_RESULT3_HASHES_MATCHED_IN_INPUT,
                                    FIELD_RESULT3_INPUT_CONFIDENCE, FIELD_RESULT3_INPUT_TOTAL_HASHES,
                                    FIELD_RESULT3_OFFSET, FIELD_RESULT3_OFFSET_SECONDS,
                                    FIELD_RESULT3_SONG_ID, FIELD_RESULT3_SONG_NAME,

                                    RESULTS_TABLENAME)


CREATE_RESULTS_TABLE_CLICKHOUSE = f"""
    CREATE TABLE IF NOT EXISTS `{RESULTS_TABLENAME}` (
    `{FIELD_RESULT_ID}` UUID DEFAULT generateUUIDv4() NOT NULL,
    `{FIELD_RESULT_TOKEN}` String NOT NULL,
    `{FIELD_RESULT1_BLOB_SHA1}` FixedString(40) NOT NULL,
    `{FIELD_RESULT1_FINGERPRINTED_CONFIDENCE}` Float64 NOT NULL,
    `{FIELD_RESULT1_FINGERPRINTED_HASHES_IN_DB}` UInt32 NOT NULL,
    `{FIELD_RESULT1_HASHES_MATCHED_IN_INPUT}` UInt32 NOT NULL,
    `{FIELD_RESULT1_INPUT_CONFIDENCE}` Float64 NOT NULL,
    `{FIELD_RESULT1_INPUT_TOTAL_HASHES}` UInt32 NOT NULL,
    `{FIELD_RESULT1_OFFSET}` UInt32 NOT NULL,
    `{FIELD_RESULT1_OFFSET_SECONDS}` Float64 NOT NULL,
    `{FIELD_RESULT1_SONG_ID}` UUID NOT NULL,
    `{FIELD_RESULT1_SONG_NAME}` String NOT NULL,

    `{FIELD_RESULT2_BLOB_SHA1}` FixedString(40),
    `{FIELD_RESULT2_FINGERPRINTED_CONFIDENCE}` Float64,
    `{FIELD_RESULT2_FINGERPRINTED_HASHES_IN_DB}` UInt32,
    `{FIELD_RESULT2_HASHES_MATCHED_IN_INPUT}` UInt32,
    `{FIELD_RESULT2_INPUT_CONFIDENCE}` Float64,
    `{FIELD_RESULT2_INPUT_TOTAL_HASHES}` UInt32,
    `{FIELD_RESULT2_OFFSET}` UInt32,
    `{FIELD_RESULT2_OFFSET_SECONDS}` Float64,
    `{FIELD_RESULT2_SONG_ID}` UUID,
    `{FIELD_RESULT2_SONG_NAME}` String,

    `{FIELD_RESULT3_BLOB_SHA1}` FixedString(40),
    `{FIELD_RESULT3_FINGERPRINTED_CONFIDENCE}` Float64,
    `{FIELD_RESULT3_FINGERPRINTED_HASHES_IN_DB}` UInt32,
    `{FIELD_RESULT3_HASHES_MATCHED_IN_INPUT}` UInt32,
    `{FIELD_RESULT3_INPUT_CONFIDENCE}` Float64,
    `{FIELD_RESULT3_INPUT_TOTAL_HASHES}` UInt32,
    `{FIELD_RESULT3_OFFSET}` UInt32,
    `{FIELD_RESULT3_OFFSET_SECONDS}` Float64,
    `{FIELD_RESULT3_SONG_ID}` UUID,
    `{FIELD_RESULT3_SONG_NAME}` String,

    `date_created` DateTime DEFAULT now() NOT NULL
    ) ENGINE = ReplacingMergeTree(date_created)
    ORDER BY `{FIELD_RESULT_TOKEN}`;
"""

CREATE_RESULTS_TABLE_MYSQL = f"""
    CREATE TABLE IF NOT EXISTS `{RESULTS_TABLENAME}` (
        `{FIELD_RESULT_ID}` VARCHAR(36) NOT NULL DEFAULT (UUID())
    ,   `{FIELD_RESULT_TOKEN}` VARCHAR(25) NOT NULL
    ,   `{FIELD_RESULT1_BLOB_SHA1}` BINARY(20) NOT NULL
    ,   `{FIELD_RESULT1_FINGERPRINTED_CONFIDENCE}` DOUBLE NOT NULL
    ,   `{FIELD_RESULT1_FINGERPRINTED_HASHES_IN_DB}` INT UNSIGNED NOT NULL
    ,   `{FIELD_RESULT1_HASHES_MATCHED_IN_INPUT}` INT UNSIGNED NOT NULL
    ,   `{FIELD_RESULT1_INPUT_CONFIDENCE}` DOUBLE NOT NULL
    ,   `{FIELD_RESULT1_INPUT_TOTAL_HASHES}` INT UNSIGNED NOT NULL
    ,   `{FIELD_RESULT1_OFFSET}` INT UNSIGNED NOT NULL
    ,   `{FIELD_RESULT1_OFFSET_SECONDS}` DOUBLE NOT NULL
    ,   `{FIELD_RESULT1_SONG_ID}` VARCHAR(36) NOT NULL
    ,   `{FIELD_RESULT1_SONG_NAME}` VARCHAR(250) NOT NULL

    ,   `{FIELD_RESULT2_BLOB_SHA1}` BINARY(20)
    ,   `{FIELD_RESULT2_FINGERPRINTED_CONFIDENCE}` DOUBLE
    ,   `{FIELD_RESULT2_FINGERPRINTED_HASHES_IN_DB}` INT UNSIGNED
    ,   `{FIELD_RESULT2_HASHES_MATCHED_IN_INPUT}` INT UNSIGNED
    ,   `{FIELD_RESULT2_INPUT_CONFIDENCE}` DOUBLE
    ,   `{FIELD_RESULT2_INPUT_TOTAL_HASHES}` INT UNSIGNED
    ,   `{FIELD_RESULT2_OFFSET}` INT UNSIGNED
    ,   `{FIELD_RESULT2_OFFSET_SECONDS}` DOUBLE
    ,   `{FIELD_RESULT2_SONG_ID}` VARCHAR(36)
    ,   `{FIELD_RESULT2_SONG_NAME}` VARCHAR(250)

    ,   `{FIELD_RESULT3_BLOB_SHA1}` BINARY(20)
    ,   `{FIELD_RESULT3_FINGERPRINTED_CONFIDENCE}` DOUBLE
    ,   `{FIELD_RESULT3_FINGERPRINTED_HASHES_IN_DB}` INT UNSIGNED
    ,   `{FIELD_RESULT3_HASHES_MATCHED_IN_INPUT}` INT UNSIGNED
    ,   `{FIELD_RESULT3_INPUT_CONFIDENCE}` DOUBLE
    ,   `{FIELD_RESULT3_INPUT_TOTAL_HASHES}` INT UNSIGNED
    ,   `{FIELD_RESULT3_OFFSET}` INT UNSIGNED
    ,   `{FIELD_RESULT3_OFFSET_SECONDS}` DOUBLE
    ,   `{FIELD_RESULT3_SONG_ID}` VARCHAR(36)
    ,   `{FIELD_RESULT3_SONG_NAME}` VARCHAR(250)

    ,   `date_created` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP

    ,   PRIMARY KEY (`{FIELD_RESULT_ID}`)
    ,   UNIQUE (`{FIELD_RESULT_TOKEN}`)
    ) ENGINE=INNODB;
"""
config_file = CONFIG_FILE if CONFIG_FILE not in [None, '', 'config.json'] else 'config.json'

def create_mysql_connection(db_credential):
    try:
        # Connect to the MySQL database
        connection = mysql.connector.connect(
            host= db_credential["host"],
            user= db_credential["user"],
            password= db_credential["password"],
            database= db_credential["database"],
            port= db_credential["port"] if db_credential.get("port") else 3306
        )
        if connection.is_connected():
            return connection
    except Exception as e:
        sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")
        return None

def create_clickhouse_connection(db_credential):
    try:
        connection = Client(
            host= db_credential["host"],
            user= db_credential["user"],
            password= db_credential["password"],
            database= db_credential["database"],
            port= db_credential["port"] if db_credential.get("port") else 9000
        )
        if connection:
            return connection
    except Exception as e:
        sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")
        return None

def if_result_token_exist(results_token, db_config, result_queue):
    if db_config["database_type"] == "clickhouse":
        clickhouse_db_connection = create_clickhouse_connection(db_config["database"])
        if clickhouse_db_connection:
            try:
                query = f"""
                SELECT `{FIELD_RESULT_ID}` FROM `{RESULTS_TABLENAME}`
                WHERE `{FIELD_RESULT_TOKEN}` = '{results_token}';
                """
                result = clickhouse_db_connection.execute(query)
                clickhouse_db_connection.disconnect()
            except Exception as e:
                sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")
                return
            if result:
                result_queue.put(result)
    elif db_config["database_type"] == "mysql":
        mysql_db_connection = create_mysql_connection(db_config["database"])
        if mysql_db_connection:
            try:
                query = f"""
                SELECT `{FIELD_RESULT_ID}` FROM `{RESULTS_TABLENAME}`
                WHERE `{FIELD_RESULT_TOKEN}` = '{results_token}';
                """
                cursor = mysql_db_connection.cursor()
                cursor.execute(query)
                result = cursor.fetchone()
                cursor.close()
                mysql_db_connection.close()
            except Exception as e:
                sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")
                return
            if result:
                result_queue.put(result[0])

def if_result_token_exist_all(results_token):
    with open(config_file, 'r') as json_file:
        db_configs = json.load(json_file)["results"]
        processes = []
        result_queue = mp.Queue()

        # Start a new process for each instance
        for item in db_configs:
            process = mp.Process(target=if_result_token_exist, args=(results_token, item, result_queue,))
            processes.append(process)
            process.start()

        # Wait for all process to complete
        for process in processes:
            process.join()

        # Collect results from each instance and merge them
        while not result_queue.empty():
            return True
        
        return False
        

def init_storage_db(db_config):
    # Initialize storage database
    if db_config["database_type"] == "clickhouse":
        try:
            clickhouse_db_connection = create_clickhouse_connection(db_config["database"])
            if clickhouse_db_connection:
                # Initalize database tables
                clickhouse_db_connection.execute(CREATE_RESULTS_TABLE_CLICKHOUSE)
                clickhouse_db_connection.disconnect()
        except Exception as e:
            sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")

    if db_config["database_type"] == "mysql":
        try:
            mysql_db_connection = create_mysql_connection(db_config["database"])
            if mysql_db_connection:
                # Initalize database tables
                cursor = mysql_db_connection.cursor()
                cursor.execute(CREATE_RESULTS_TABLE_MYSQL)
                cursor.close()
                mysql_db_connection.close()
        except Exception as e:
                sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")

def init_all_storage_db():
    # Initialize all storage database with multi processing
    with open(config_file, 'r') as json_file:
        db_configs = json.load(json_file)["results"]
        processes = []

        # Start a new process for each instance
        for item in db_configs:
            process = mp.Process(target=init_storage_db, args=(item,))
            processes.append(process)
            process.start()

        # Wait for all process to complete
        for process in processes:
            process.join()

def store_result(results_token, results_array):
    try:
        with open(config_file, 'r') as json_file:
            db_configs = json.load(json_file)["results"]
            selected_db = select_database(db_configs, RESULTS_TABLENAME)
            if selected_db["database_type"] == "clickhouse":
                clickhouse_db_connection = create_clickhouse_connection(selected_db["database"])
                store_result_base_query = f"""
                    INSERT INTO `{RESULTS_TABLENAME}`(
                        `{FIELD_RESULT_TOKEN}`,
                        `{FIELD_RESULT1_BLOB_SHA1}`, `{FIELD_RESULT1_FINGERPRINTED_CONFIDENCE}`, `{FIELD_RESULT1_FINGERPRINTED_HASHES_IN_DB}`,
                        `{FIELD_RESULT1_HASHES_MATCHED_IN_INPUT}`, `{FIELD_RESULT1_INPUT_CONFIDENCE}`, `{FIELD_RESULT1_INPUT_TOTAL_HASHES}`,
                        `{FIELD_RESULT1_OFFSET}`, `{FIELD_RESULT1_OFFSET_SECONDS}`, `{FIELD_RESULT1_SONG_ID}`, `{FIELD_RESULT1_SONG_NAME}`,

                        `{FIELD_RESULT2_BLOB_SHA1}`, `{FIELD_RESULT2_FINGERPRINTED_CONFIDENCE}`, `{FIELD_RESULT2_FINGERPRINTED_HASHES_IN_DB}`,
                        `{FIELD_RESULT2_HASHES_MATCHED_IN_INPUT}`, `{FIELD_RESULT2_INPUT_CONFIDENCE}`, `{FIELD_RESULT2_INPUT_TOTAL_HASHES}`,
                        `{FIELD_RESULT2_OFFSET}`, `{FIELD_RESULT2_OFFSET_SECONDS}`, `{FIELD_RESULT2_SONG_ID}`, `{FIELD_RESULT2_SONG_NAME}`,
                        
                        `{FIELD_RESULT3_BLOB_SHA1}`, `{FIELD_RESULT3_FINGERPRINTED_CONFIDENCE}`, `{FIELD_RESULT3_FINGERPRINTED_HASHES_IN_DB}`,
                        `{FIELD_RESULT3_HASHES_MATCHED_IN_INPUT}`, `{FIELD_RESULT3_INPUT_CONFIDENCE}`, `{FIELD_RESULT3_INPUT_TOTAL_HASHES}`,
                        `{FIELD_RESULT3_OFFSET}`, `{FIELD_RESULT3_OFFSET_SECONDS}`, `{FIELD_RESULT3_SONG_ID}`, `{FIELD_RESULT3_SONG_NAME}`
                        )
                        VALUES
                    """
                if len(results_array) > 2 and results_array[1] and results_array[2]:
                    store_result_dataset = f"""
                    (
                        '{results_token}',
                        '{results_array[0][FIELD_BLOB_SHA1]}', '{results_array[0][FINGERPRINTED_CONFIDENCE]}', '{results_array[0][FINGERPRINTED_HASHES]}',
                        '{results_array[0][HASHES_MATCHED]}', '{results_array[0][INPUT_CONFIDENCE]}', '{results_array[0][INPUT_HASHES]}',
                        '{results_array[0][FIELD_OFFSET]}', '{results_array[0][OFFSET_SECS]}', '{results_array[0][FIELD_SONG_ID]}', '{results_array[0][FIELD_SONGNAME]}',

                        '{results_array[1][FIELD_BLOB_SHA1]}', '{results_array[1][FINGERPRINTED_CONFIDENCE]}', '{results_array[1][FINGERPRINTED_HASHES]}',
                        '{results_array[1][HASHES_MATCHED]}', '{results_array[1][INPUT_CONFIDENCE]}', '{results_array[1][INPUT_HASHES]}',
                        '{results_array[1][FIELD_OFFSET]}', '{results_array[1][OFFSET_SECS]}', '{results_array[1][FIELD_SONG_ID]}', '{results_array[1][FIELD_SONGNAME]}',

                        '{results_array[2][FIELD_BLOB_SHA1]}', '{results_array[2][FINGERPRINTED_CONFIDENCE]}', '{results_array[2][FINGERPRINTED_HASHES]}',
                        '{results_array[2][HASHES_MATCHED]}', '{results_array[2][INPUT_CONFIDENCE]}', '{results_array[2][INPUT_HASHES]}',
                        '{results_array[2][FIELD_OFFSET]}', '{results_array[2][OFFSET_SECS]}', '{results_array[2][FIELD_SONG_ID]}', '{results_array[2][FIELD_SONGNAME]}'
                    )
                    """
                    clickhouse_db_connection.execute(store_result_base_query + " " + store_result_dataset)
                    clickhouse_db_connection.disconnect()
                    return 0
                elif len(results_array) == 2:
                    store_result_dataset = f"""
                    (
                        '{results_token}',
                        '{results_array[0][FIELD_BLOB_SHA1]}', '{results_array[0][FINGERPRINTED_CONFIDENCE]}', '{results_array[0][FINGERPRINTED_HASHES]}',
                        '{results_array[0][HASHES_MATCHED]}', '{results_array[0][INPUT_CONFIDENCE]}', '{results_array[0][INPUT_HASHES]}',
                        '{results_array[0][FIELD_OFFSET]}', '{results_array[0][OFFSET_SECS]}', '{results_array[0][FIELD_SONG_ID]}', '{results_array[0][FIELD_SONGNAME]}',

                        '{results_array[1][FIELD_BLOB_SHA1]}', '{results_array[1][FINGERPRINTED_CONFIDENCE]}', '{results_array[1][FINGERPRINTED_HASHES]}',
                        '{results_array[1][HASHES_MATCHED]}', '{results_array[1][INPUT_CONFIDENCE]}', '{results_array[1][INPUT_HASHES]}',
                        '{results_array[1][FIELD_OFFSET]}', '{results_array[1][OFFSET_SECS]}', '{results_array[1][FIELD_SONG_ID]}', '{results_array[1][FIELD_SONGNAME]}',

                        NULL, NULL, NULL,
                        NULL, NULL, NULL,
                        NULL, NULL, NULL, NULL
                    )
                    """
                    clickhouse_db_connection.execute(store_result_base_query + " " + store_result_dataset)
                    clickhouse_db_connection.disconnect()
                    return 0
                elif len(results_array) == 1:
                    store_result_dataset = f"""
                    (
                        '{results_token}',
                        '{results_array[0][FIELD_BLOB_SHA1]}', '{results_array[0][FINGERPRINTED_CONFIDENCE]}', '{results_array[0][FINGERPRINTED_HASHES]}',
                        '{results_array[0][HASHES_MATCHED]}', '{results_array[0][INPUT_CONFIDENCE]}', '{results_array[0][INPUT_HASHES]}',
                        '{results_array[0][FIELD_OFFSET]}', '{results_array[0][OFFSET_SECS]}', '{results_array[0][FIELD_SONG_ID]}', '{results_array[0][FIELD_SONGNAME]}',
                        
                        NULL, NULL, NULL,
                        NULL, NULL, NULL,
                        NULL, NULL, NULL, NULL,

                        NULL, NULL, NULL,
                        NULL, NULL, NULL,
                        NULL, NULL, NULL, NULL
                    )
                    """
                    clickhouse_db_connection.execute(store_result_base_query + " " + store_result_dataset)
                    clickhouse_db_connection.disconnect()
                    return 0
                else:
                    clickhouse_db_connection.disconnect()
                    return 1

            if selected_db["database_type"] == "mysql":
                mysql_db_connection = create_mysql_connection(selected_db["database"])
                cursor = mysql_db_connection.cursor()
                store_result = f"""
                    INSERT INTO `{RESULTS_TABLENAME}`(
                    `{FIELD_RESULT_TOKEN}`,
                    `{FIELD_RESULT1_BLOB_SHA1}`, `{FIELD_RESULT1_FINGERPRINTED_CONFIDENCE}`, `{FIELD_RESULT1_FINGERPRINTED_HASHES_IN_DB}`,
                    `{FIELD_RESULT1_HASHES_MATCHED_IN_INPUT}`, `{FIELD_RESULT1_INPUT_CONFIDENCE}`, `{FIELD_RESULT1_INPUT_TOTAL_HASHES}`,
                    `{FIELD_RESULT1_OFFSET}`, `{FIELD_RESULT1_OFFSET_SECONDS}`, `{FIELD_RESULT1_SONG_ID}`, `{FIELD_RESULT1_SONG_NAME}`,

                    `{FIELD_RESULT2_BLOB_SHA1}`, `{FIELD_RESULT2_FINGERPRINTED_CONFIDENCE}`, `{FIELD_RESULT2_FINGERPRINTED_HASHES_IN_DB}`,
                    `{FIELD_RESULT2_HASHES_MATCHED_IN_INPUT}`, `{FIELD_RESULT2_INPUT_CONFIDENCE}`, `{FIELD_RESULT2_INPUT_TOTAL_HASHES}`,
                    `{FIELD_RESULT2_OFFSET}`, `{FIELD_RESULT2_OFFSET_SECONDS}`, `{FIELD_RESULT2_SONG_ID}`, `{FIELD_RESULT2_SONG_NAME}`,
                    
                    `{FIELD_RESULT3_BLOB_SHA1}`, `{FIELD_RESULT3_FINGERPRINTED_CONFIDENCE}`, `{FIELD_RESULT3_FINGERPRINTED_HASHES_IN_DB}`,
                    `{FIELD_RESULT3_HASHES_MATCHED_IN_INPUT}`, `{FIELD_RESULT3_INPUT_CONFIDENCE}`, `{FIELD_RESULT3_INPUT_TOTAL_HASHES}`,
                    `{FIELD_RESULT3_OFFSET}`, `{FIELD_RESULT3_OFFSET_SECONDS}`, `{FIELD_RESULT3_SONG_ID}`, `{FIELD_RESULT3_SONG_NAME}`
                    )
                    VALUES (
                    %s,
                    UNHEX(%s), %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,

                    UNHEX(%s), %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,

                    UNHEX(%s), %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s
                    );
                """
                values_to_insert = []
                if len(results_array) > 2 and results_array[1] and results_array[2]:
                    values_to_insert.append((
                        results_token,
                        results_array[0][FIELD_BLOB_SHA1], results_array[0][FINGERPRINTED_CONFIDENCE], results_array[0][FINGERPRINTED_HASHES],
                        results_array[0][HASHES_MATCHED], results_array[0][INPUT_CONFIDENCE], results_array[0][INPUT_HASHES],
                        results_array[0][FIELD_OFFSET], results_array[0][OFFSET_SECS], results_array[0][FIELD_SONG_ID], results_array[0][FIELD_SONGNAME],

                        results_array[1][FIELD_BLOB_SHA1], results_array[1][FINGERPRINTED_CONFIDENCE], results_array[1][FINGERPRINTED_HASHES],
                        results_array[1][HASHES_MATCHED], results_array[1][INPUT_CONFIDENCE], results_array[1][INPUT_HASHES],
                        results_array[1][FIELD_OFFSET], results_array[1][OFFSET_SECS], results_array[1][FIELD_SONG_ID], results_array[1][FIELD_SONGNAME],

                        results_array[2][FIELD_BLOB_SHA1], results_array[2][FINGERPRINTED_CONFIDENCE], results_array[2][FINGERPRINTED_HASHES],
                        results_array[2][HASHES_MATCHED], results_array[2][INPUT_CONFIDENCE], results_array[2][INPUT_HASHES],
                        results_array[2][FIELD_OFFSET], results_array[2][OFFSET_SECS], results_array[2][FIELD_SONG_ID], results_array[2][FIELD_SONGNAME]
                    ))
                elif len(results_array) == 2:
                    values_to_insert.append((
                        results_token,
                        results_array[0][FIELD_BLOB_SHA1], results_array[0][FINGERPRINTED_CONFIDENCE], results_array[0][FINGERPRINTED_HASHES],
                        results_array[0][HASHES_MATCHED], results_array[0][INPUT_CONFIDENCE], results_array[0][INPUT_HASHES],
                        results_array[0][FIELD_OFFSET], results_array[0][OFFSET_SECS], results_array[0][FIELD_SONG_ID], results_array[0][FIELD_SONGNAME],

                        results_array[1][FIELD_BLOB_SHA1], results_array[1][FINGERPRINTED_CONFIDENCE], results_array[1][FINGERPRINTED_HASHES],
                        results_array[1][HASHES_MATCHED], results_array[1][INPUT_CONFIDENCE], results_array[1][INPUT_HASHES],
                        results_array[1][FIELD_OFFSET], results_array[1][OFFSET_SECS], results_array[1][FIELD_SONG_ID], results_array[1][FIELD_SONGNAME],

                        None, None, None,
                        None, None, None,
                        None, None, None, None
                    ))
                elif len(results_array) == 1:
                    values_to_insert.append((
                        results_token,
                        results_array[0][FIELD_BLOB_SHA1], results_array[0][FINGERPRINTED_CONFIDENCE], results_array[0][FINGERPRINTED_HASHES],
                        results_array[0][HASHES_MATCHED], results_array[0][INPUT_CONFIDENCE], results_array[0][INPUT_HASHES],
                        results_array[0][FIELD_OFFSET], results_array[0][OFFSET_SECS], results_array[0][FIELD_SONG_ID], results_array[0][FIELD_SONGNAME],

                        None, None, None,
                        None, None, None,
                        None, None, None, None,

                        None, None, None,
                        None, None, None,
                        None, None, None, None
                    ))
                else:  # When the result set is empty
                    cursor.close()
                    mysql_db_connection.close()
                    return 1 
                
                cursor.executemany(store_result, values_to_insert)
                mysql_db_connection.commit()
                cursor.close()
                mysql_db_connection.close()
                return 0 # when the result is successfully stored in database
                
    except Exception as e: # when error occurs
        sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")
        return 2

def search_result_in_db(results_token, db_config, result_queue):
    if db_config["database_type"] == "clickhouse":
        try:
            clickhouse_db_connection = create_clickhouse_connection(db_config["database"])
            search_query = f"""
            SELECT
                `{FIELD_RESULT_TOKEN}`,
                `{FIELD_RESULT1_BLOB_SHA1}`, `{FIELD_RESULT1_FINGERPRINTED_CONFIDENCE}`, `{FIELD_RESULT1_FINGERPRINTED_HASHES_IN_DB}`,
                `{FIELD_RESULT1_HASHES_MATCHED_IN_INPUT}`, `{FIELD_RESULT1_INPUT_CONFIDENCE}`, `{FIELD_RESULT1_INPUT_TOTAL_HASHES}`,
                `{FIELD_RESULT1_OFFSET}`, `{FIELD_RESULT1_OFFSET_SECONDS}`, `{FIELD_RESULT1_SONG_ID}`, `{FIELD_RESULT1_SONG_NAME}`,

                `{FIELD_RESULT2_BLOB_SHA1}`, `{FIELD_RESULT2_FINGERPRINTED_CONFIDENCE}`, `{FIELD_RESULT2_FINGERPRINTED_HASHES_IN_DB}`,
                `{FIELD_RESULT2_HASHES_MATCHED_IN_INPUT}`, `{FIELD_RESULT2_INPUT_CONFIDENCE}`, `{FIELD_RESULT2_INPUT_TOTAL_HASHES}`,
                `{FIELD_RESULT2_OFFSET}`, `{FIELD_RESULT2_OFFSET_SECONDS}`, `{FIELD_RESULT2_SONG_ID}`, `{FIELD_RESULT2_SONG_NAME}`,
                    
                `{FIELD_RESULT3_BLOB_SHA1}`, `{FIELD_RESULT3_FINGERPRINTED_CONFIDENCE}`, `{FIELD_RESULT3_FINGERPRINTED_HASHES_IN_DB}`,
                `{FIELD_RESULT3_HASHES_MATCHED_IN_INPUT}`, `{FIELD_RESULT3_INPUT_CONFIDENCE}`, `{FIELD_RESULT3_INPUT_TOTAL_HASHES}`,
                `{FIELD_RESULT3_OFFSET}`, `{FIELD_RESULT3_OFFSET_SECONDS}`, `{FIELD_RESULT3_SONG_ID}`, `{FIELD_RESULT3_SONG_NAME}`
            
            FROM `{RESULTS_TABLENAME}`
            WHERE `{FIELD_RESULT_TOKEN}` = '{results_token}';
            """
            result = clickhouse_db_connection.execute(search_query)
            if result:
                result_queue.put(result[0])
            clickhouse_db_connection.disconnect()

        except Exception as e:
            sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")

    if db_config["database_type"] == "mysql":
        try:
            mysql_db_connection = create_mysql_connection(db_config["database"])
            cursor = mysql_db_connection.cursor()
            search_query = f"""
            SELECT
                `{FIELD_RESULT_TOKEN}`,
                `{FIELD_RESULT1_BLOB_SHA1}`, `{FIELD_RESULT1_FINGERPRINTED_CONFIDENCE}`, `{FIELD_RESULT1_FINGERPRINTED_HASHES_IN_DB}`,
                `{FIELD_RESULT1_HASHES_MATCHED_IN_INPUT}`, `{FIELD_RESULT1_INPUT_CONFIDENCE}`, `{FIELD_RESULT1_INPUT_TOTAL_HASHES}`,
                `{FIELD_RESULT1_OFFSET}`, `{FIELD_RESULT1_OFFSET_SECONDS}`, `{FIELD_RESULT1_SONG_ID}`, `{FIELD_RESULT1_SONG_NAME}`,

                `{FIELD_RESULT2_BLOB_SHA1}`, `{FIELD_RESULT2_FINGERPRINTED_CONFIDENCE}`, `{FIELD_RESULT2_FINGERPRINTED_HASHES_IN_DB}`,
                `{FIELD_RESULT2_HASHES_MATCHED_IN_INPUT}`, `{FIELD_RESULT2_INPUT_CONFIDENCE}`, `{FIELD_RESULT2_INPUT_TOTAL_HASHES}`,
                `{FIELD_RESULT2_OFFSET}`, `{FIELD_RESULT2_OFFSET_SECONDS}`, `{FIELD_RESULT2_SONG_ID}`, `{FIELD_RESULT2_SONG_NAME}`,
                    
                `{FIELD_RESULT3_BLOB_SHA1}`, `{FIELD_RESULT3_FINGERPRINTED_CONFIDENCE}`, `{FIELD_RESULT3_FINGERPRINTED_HASHES_IN_DB}`,
                `{FIELD_RESULT3_HASHES_MATCHED_IN_INPUT}`, `{FIELD_RESULT3_INPUT_CONFIDENCE}`, `{FIELD_RESULT3_INPUT_TOTAL_HASHES}`,
                `{FIELD_RESULT3_OFFSET}`, `{FIELD_RESULT3_OFFSET_SECONDS}`, `{FIELD_RESULT3_SONG_ID}`, `{FIELD_RESULT3_SONG_NAME}`
            
            FROM `{RESULTS_TABLENAME}`
            WHERE `{FIELD_RESULT_TOKEN}` = %s;
            """
            cursor.execute(search_query, (results_token,))
            result = cursor.fetchone()
            cursor.close()
            mysql_db_connection.close()
            result_queue.put(result)
        except Exception as e:
            sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")

def search_result_all(results_token):
    with open(config_file, 'r') as json_file:
        db_configs = json.load(json_file)["results"]

        processes = []
        result_queue = mp.Queue()

        # Start a new process for each instance
        for item in db_configs:
            process = mp.Process(target=search_result_in_db, args=(results_token, item, result_queue,))
            processes.append(process)
            process.start()

        # Wait for all process to complete
        for process in processes:
            process.join()

        # Collect results from each instance and merge them
        while not result_queue.empty():
            result = result_queue.get()
            if result:
                binary_json = {
                    "results": [],
                    "token": result[0],
                    "status": "success"
                }
                if result[1]:
                    binary_json["results"].append({
                        FIELD_BLOB_SHA1 : result[1] if type(result[1]) is str else result[1].hex(),
                        FINGERPRINTED_CONFIDENCE: result[2],
                        FINGERPRINTED_HASHES: result[3],
                        HASHES_MATCHED: result[4],
                        INPUT_CONFIDENCE: result[5],
                        INPUT_HASHES: result[6],
                        FIELD_OFFSET: result[7],
                        OFFSET_SECS: result[8],
                        FIELD_SONG_ID: result[9],
                        FIELD_SONGNAME: result[10]
                    })
                if result[11]:
                    binary_json["results"].append({
                        FIELD_BLOB_SHA1: result[11] if type(result[11]) is str else result[1].hex(),
                        FINGERPRINTED_CONFIDENCE: result[12],
                        FINGERPRINTED_HASHES: result[13],
                        HASHES_MATCHED: result[14],
                        INPUT_CONFIDENCE: result[15],
                        INPUT_HASHES: result[16],
                        FIELD_OFFSET: result[17],
                        OFFSET_SECS: result[18],
                        FIELD_SONG_ID: result[19],
                        FIELD_SONGNAME: result[20]
                    })
                if result[21]:
                    binary_json["results"].append({
                        FIELD_BLOB_SHA1: result[21] if type(result[21]) is str else result[1].hex(),
                        FINGERPRINTED_CONFIDENCE: result[22],
                        FINGERPRINTED_HASHES: result[23],
                        HASHES_MATCHED: result[24],
                        INPUT_CONFIDENCE: result[25],
                        INPUT_HASHES: result[26],
                        FIELD_OFFSET: result[27],
                        OFFSET_SECS: result[28],
                        FIELD_SONG_ID: result[29],
                        FIELD_SONGNAME: result[30]
                    })
                # Make sure that the returned data format is JSON compatible
                json_result = jsonify_binary(binary_json)
                return json_result