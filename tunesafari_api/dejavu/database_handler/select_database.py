import multiprocessing as mp
from clickhouse_driver import Client
import mysql.connector
from mysql.connector import Error
import sys

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
    except Error as e:
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
    except Error as e:
        sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")
        return None

def check_database_record(db_config, table_name, result_queue):
    if db_config["database_type"] == "clickhouse":
        clickhouse_db_connection = create_clickhouse_connection(db_config["database"])
        if clickhouse_db_connection:
            try:
                db_record_count_query = f"""
                SELECT COUNT(*) FROM `{table_name}`;
                """
                result = clickhouse_db_connection.execute(db_record_count_query)
                if result:
                    result_queue.put((result[0], db_config))
            except Exception as e:
                sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")
                return None

    elif db_config["database_type"] == "mysql":
        mysql_db_connection = create_mysql_connection(db_config["database"])
        if mysql_db_connection:
            try:
                db_record_count_query = f"""
                SELECT COUNT(*) FROM `{table_name}`;
                """
                cursor = mysql_db_connection.cursor()
                cursor.execute(db_record_count_query)
                result = cursor.fetchone()
                if result:
                    result_queue.put((result[0], db_config)) # Turple (record_count, database_config)
            except Exception as e:
                sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")
                return None

# Select the database with the least records
def select_database(db_configs, table_name):
    processes = []
    result_queue = mp.Queue()

    for instance in db_configs:
        process = mp.Process(target=check_database_record, args=(instance, table_name, result_queue,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    least_records_db = result_queue.get() # Get the first turple (record_count, database_config)

    while not result_queue.empty():
        result = result_queue.get()
        if least_records_db[0] > result[0]: # Update the (record_count, database_config) turple with the least record
            least_records_db = result

    return least_records_db[1] # Return the database config with the least record