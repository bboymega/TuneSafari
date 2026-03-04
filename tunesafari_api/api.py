from dejavu.core_modules.recognize_from_api import recognize_all
from dejavu.core_modules.fingerprint_from_api import fingerprint
from flask import Flask, jsonify, request, abort
import json
import secrets
import string
from datetime import datetime
from dejavu import Dejavu
from dejavu.base_classes.jsonify_binary_data import jsonify_binary
from dejavu.config.settings import (CONFIG_FILE, DEFAULT_FS, HASHES_MATCHED, FINGERPRINTED_CONFIDENCE)
from dejavu.database_handler.result_storage import init_all_storage_db, store_result, search_result_all, if_result_token_exist_all
from pathlib import Path
from io import BytesIO
from werkzeug.middleware.proxy_fix import ProxyFix
import ffmpeg
import sys
import re
import random
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import traceback
import tempfile
import os

config_file = CONFIG_FILE if CONFIG_FILE not in [None, '', 'config.json'] else 'config.json'

def create_app():
    app = Flask(__name__)
    with open(config_file, "r") as f:
        config = json.load(f)
        if(config.get("allowed_origin")):
            allowed_origin = config.get("allowed_origin")
            CORS(app, resources={r"/*": {"origins": allowed_origin}})
        app.wsgi_app = ProxyFix(
        app.wsgi_app, 
        x_for=1, 
        x_proto=1, 
        x_host=1, 
        x_prefix=1
    )
    return app

app = create_app()

with open('config.json') as f:
    config_data = json.load(f)

raw_recognize_limit = config_data.get("rate_limit", {}).get("recognizing")
active_recognize_limit = raw_recognize_limit if raw_recognize_limit else "10 per second"

raw_fetch_limit = config_data.get("rate_limit", {}).get("fetching_results")
active_fetch_limit = raw_fetch_limit if raw_fetch_limit else "50 per second"

raw_fingerprint_limit = config_data.get("rate_limit", {}).get("fingerprinting")
active_fingerprint_limit = raw_fingerprint_limit if raw_fingerprint_limit else "10 per second"

try:
    with open(config_file, 'r') as f:
        config_data = json.load(f)

    redis_conf = config_data.get("redis", {})
    host = redis_conf.get("host", "127.0.0.1")
    user = redis_conf.get("user", "")
    password = redis_conf.get("password", "")
    port = redis_conf.get("port", 6379)
    prefix = redis_conf.get("prefix", "TuneSafari")
    db_index = config_data.get("rate_limit", {}).get("redis_db_index", random.randint(0, 15))
    limiter = Limiter(
        get_remote_address,
        app=app,
        storage_uri=f"redis://{user}:{password}@{host}:{port}/{db_index}",
        storage_options={"key_prefix": prefix}
    )
except Exception as e:
    sys.stderr.write(f"\033[33m{datetime.now().strftime("[%d/%b/%Y %H:%M:%S]")} TuneSafari \"WARNING: Redis Connection Warning: {e}. Falling back to in-memory mode for rate limiting\"\033[0m\n")
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        storage_uri="memory://"
    )

def init():
    try:
        print(f"{datetime.now().strftime("[%d/%b/%Y %H:%M:%S]")} TuneSafari \"INFO: Initializing TuneSafari\"", flush=True)
        init_all_storage_db()
        with open(config_file) as f:
            config = json.load(f)
            instances = []
            for item in config["instances"]:
                try:
                    djv_item = Dejavu(item)
                    if djv_item:
                        instances.append(djv_item)
                except Exception as e:
                    sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")
        print(f"{datetime.now().strftime("[%d/%b/%Y %H:%M:%S]")} TuneSafari \"INFO: worker is ready to accept requests\"", flush=True)
    except Exception as e:
        sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")


def generate_result_token(length=20):
    # Define the characters to choose from (uppercase, lowercase, digits)
    characters = string.ascii_letters + string.digits
    # Generate a random string using secrets.choice
    random_id = ''.join(secrets.choice(characters) for _ in range(length))
    return random_id

def sanitize_filename(filename):
    # Remove leading/trailing spaces
    filename = filename.strip()
    # Replace any special characters with underscores
    # This pattern covers common special characters that might be unsafe
    invalid_chars = r'[<>:"/\\|?*`!@#$%^&()&{}[\];+=,\'"]'
    filename = re.sub(invalid_chars, '_', filename)
    # Return the sanitized filename
    return filename

@app.route('/api/recognize', methods=['POST'])
@limiter.limit(active_recognize_limit)
def recognize_api():
    try:
        print(f"{datetime.now().strftime("[%d/%b/%Y %H:%M:%S]")} {request.remote_addr} \"INFO: Incoming request from client for recognition process\"", flush=True)

        if 'file' not in request.files:
            sys.stderr.write("\033[31m" + f"{datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} {request.remote_addr} \"ERROR: No file part in the request\"" + "\033[0m\n")
            return jsonify({
                "status": "error",
                "message": "No file part in the request"
            }), 400

        with open(config_file, 'r') as jsonFile:
            json_config = json.load(jsonFile)
            if "recognizing" in json_config:
                recognizing_config = json_config["recognizing"]
                if "max_file_size_mb" in recognizing_config:
                    if isinstance(recognizing_config["max_file_size_mb"], str):
                        max_file_size_mb = int(recognizing_config["max_file_size_mb"])
                        if request.content_length and request.content_length > max_file_size_mb * 1024 * 1024:
                            sys.stderr.write("\033[31m" + f"{datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} {request.remote_addr} \"ERROR: File exceeds the size limit {max_file_size_mb * 1024 * 1024}, received bytes {request.content_length}\"" + "\033[0m\n")
                            return jsonify({
                                "status": "error",
                                "message": f"File exceeds the {max_file_size_mb} MB limit",
                                "limit_bytes": f"{max_file_size_mb * 1024 * 1024}",
                                "received_bytes": f"{request.content_length}"
                            }), 413
                    else:
                        max_file_size_mb = recognizing_config["max_file_size_mb"]
                        if request.content_length and request.content_length > max_file_size_mb * 1024 * 1024:
                            sys.stderr.write("\033[31m" + f"{datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} {request.remote_addr} \"ERROR: File exceeds the size limit {max_file_size_mb * 1024 * 1024}, received bytes {request.content_length}\"" + "\033[0m\n")
                            return jsonify({
                                "status": "error",
                                "message": f"File exceeds the {max_file_size_mb} MB limit",
                                "limit_bytes": f"{max_file_size_mb * 1024 * 1024}",
                                "received_bytes": f"{request.content_length}"
                            }), 413

        blob = request.files['file'].read()

        # convert audio to standard wav before sampling
        convert_only = False # If the audio should be trimmed
        with open(config_file, 'r') as jsonFile:
            json_config = json.load(jsonFile)
            if "recognizing" in json_config:
                recognizing_config = json_config["recognizing"]
                if "max_duration" in recognizing_config:
                    if isinstance(recognizing_config["max_duration"], str):
                        max_duration = int(recognizing_config["max_duration"])
                    else:
                        max_duration = recognizing_config["max_duration"]

                        if request.form.get('start'):
                            start_time = float(request.form.get('start'))
                        else:
                            start_time = 0.0

                        if request.form.get('duration'):
                            duration = float(request.form.get('duration'))
                            if duration > max_duration * 1.0:
                                duration = max_duration * 1.0
                        else:
                            duration = max_duration * 1.0

                    if max_duration > 0:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as temp_data:
                            temp_data.write(blob)
                            temp_data_path = temp_data.name
                        try:
                            blob, err = ffmpeg.input(temp_data_path, ss=start_time, t=duration) \
                            .filter('loudnorm', I=-16, TP=-1.5, LRA=11) \
                            .output('pipe:1', format='wav', ar=DEFAULT_FS, ac=1, sample_fmt='s16') \
                            .run(capture_stdout=True, capture_stderr=True)
                            if not blob or len(blob) < 100:
                                sys.stderr.write("\033[31m" + "ERROR: FFmpeg returned empty audio data: " + err.decode() + "\033[0m\n")
                                return jsonify({"status": "error", "message": "Extracted audio is empty"}), 400
                        except Exception as e:
                            sys.stderr.write("\033[31m" + f"{datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} {request.remote_addr} \"ERROR: Failed to process input file {e}\"" + "\033[0m\n")
                            return jsonify({
                                "status": "error",
                                "message": "Failed to process input file"
                            }), 500
                        finally:
                            if os.path.exists(temp_data_path):
                                os.remove(temp_data_path)
                    else:
                        convert_only = True
                else:
                    convert_only = True
            else:
                convert_only = True
        if convert_only:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as temp_data:
                temp_data.write(blob)
                temp_data_path = temp_data.name
            try:
                if request.form.get('start'):
                    start_time = float(request.form.get('start'))
                else:
                    start_time = 0.0
                
                if request.form.get('duration'):
                    duration = float(request.form.get('duration'))
                    blob, err = ffmpeg.input(temp_data_path, ss=start_time, t=duration) \
                    .filter('loudnorm', I=-16, TP=-1.5, LRA=11) \
                    .output('pipe:1', format='wav', ar=DEFAULT_FS, ac=1, sample_fmt='s16') \
                    .run(capture_stdout=True, capture_stderr=True)
                    if not blob or len(blob) == 0:
                        sys.stderr.write("\033[31m" + "ERROR: FFmpeg returned empty audio data: " + err.decode() + "\033[0m\n")
                        return jsonify({"status": "error", "message": "Extracted audio is empty"}), 400
                else:
                    blob, err = ffmpeg.input(temp_data_path, ss=start_time) \
                    .filter('loudnorm', I=-16, TP=-1.5, LRA=11) \
                    .output('pipe:1', format='wav', ar=DEFAULT_FS, ac=1, sample_fmt='s16') \
                    .run(capture_stdout=True, capture_stderr=True)
                    if not blob or len(blob) < 100:
                        sys.stderr.write("\033[31m" + "ERROR: FFmpeg returned empty audio data: " + err.decode() + "\033[0m\n")
                        return jsonify({"status": "error", "message": "Extracted audio is empty"}), 400

            except Exception as e:
                sys.stderr.write("\033[31m" + f"{datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} {request.remote_addr} \"ERROR: Failed to process input file {e}\"" + "\033[0m\n")
                return jsonify({
                    "status": "error",
                    "message": "Failed to process input file"
                }), 500
            finally:
                if os.path.exists(temp_data_path):
                    os.remove(temp_data_path)
        
        results = recognize_all(blob)
        results_array = [] # Here stores the results
        for result in results:
            results_array.append(jsonify_binary(result)) # Make sure that the returned data format is JSON compatible
        
        results_array = sorted(
            results_array,
            key=lambda x: (x[HASHES_MATCHED], x[FINGERPRINTED_CONFIDENCE]),
            reverse=True
        )
        
        results_array = results_array[:3] # Only return the 3 best solutions.
        results_token = generate_result_token()

        # Make sure the result token is unique
        while if_result_token_exist_all(results_token):
            results_token = generate_result_token()
            
        result_status = store_result(results_token, results_array)
        if result_status == 0:
            print(f"{datetime.now().strftime("[%d/%b/%Y %H:%M:%S]")} {request.remote_addr} \"INFO: Recognition result generated, token: {results_token}\"", flush=True)
            return(jsonify({"token":results_token, "results": results_array, "status": "success"}))
        elif result_status == 1:
            sys.stderr.write("\033[93m" + f"{datetime.now().strftime("[%d/%b/%Y %H:%M:%S]")} {request.remote_addr} \"WARNING: No results were found\"" + "\033[0m\n")
            return jsonify({
                "status": "success",
                "results": []
            }), 200
        else:
            sys.stderr.write("\033[31m" + f"{datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} {request.remote_addr} \"ERROR: Failed to store result\"" + "\033[0m\n")
            return jsonify({
                "status": "error",
                "message": "Falied to store result"
            }), 500
    except Exception as e:
        traceback_info = traceback.format_exc()
        sys.stderr.write("\033[31m" + "\n--- Full Traceback ---" + "\033[0m\n")
        sys.stderr.write("\033[31m" + traceback_info + "\033[0m\n")
        sys.stderr.write("\033[31m----------------------\033[0m\n")
        sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")
        abort(500)

@app.route('/api/fetch/<token>', methods=['GET'])
@limiter.limit(active_fetch_limit)
def fetch_result_api(token):
    try:
        json_result = search_result_all(token)
        if json_result:
            print(f"{datetime.now().strftime("[%d/%b/%Y %H:%M:%S]")} {request.remote_addr} \"INFO: Successfully fetched result, token: {token}\"", flush=True)
            return jsonify(json_result)
        else:
            sys.stderr.write("\033[33m" + f"{datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} {request.remote_addr} \"WARNING: The requested data could not be found, token: {token}\"" + "\033[0m\n")
            return jsonify({
                "status": "error",
                "message": "The requested data could not be found"
            }), 404
    except Exception as e:
        traceback_info = traceback.format_exc()
        sys.stderr.write("\033[31m" + "\n--- Full Traceback ---" + "\033[0m\n")
        sys.stderr.write("\033[31m" + traceback_info + "\033[0m\n")
        sys.stderr.write("\033[31m----------------------\033[0m\n")
        sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")
        sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")
        abort(500)


@app.route('/api/fingerprint', methods=['POST'])
@limiter.limit(active_fingerprint_limit)
def fingerprint_api():
    try:
        if 'file' not in request.files:
            print(f"{datetime.now().strftime("[%d/%b/%Y %H:%M:%S]")} {request.remote_addr} \"INFO: Incoming request from client for fingerprinting process, filename: {None}\"", flush=True)
            sys.stderr.write("\033[31m" + f"{datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} {request.remote_addr} \"ERROR: No file part in the request\"" + "\033[0m\n")
            return jsonify({
                "status": "error",
                "message": "No file part in the request"
            }), 400
        uploaded_filename = Path(request.files["file"].filename).name
        print(f"{datetime.now().strftime("[%d/%b/%Y %H:%M:%S]")} {request.remote_addr} \"INFO: Incoming request from client for fingerprinting process, filename: {uploaded_filename}\"", flush=True)

        with open(config_file, 'r') as jsonFile:
            config_json = json.load(jsonFile)
            allow_fingerprinting = config_json["fingerprinting"]["allow"]

            # Check if fingerprinting is allowed
            if isinstance(allow_fingerprinting, str):
                allow_fingerprinting = allow_fingerprinting.strip().lower() in ['true', '1'] # Normalize data type true, 'true', '1', 1
            
            # "allow_fingerprinting": false. Write protection enabled
            if not allow_fingerprinting:
                sys.stderr.write("\033[31m" + f"{datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} {request.remote_addr} \"ERROR: Fingerprinting not allowed\"" + "\033[0m\n")
                return jsonify({
                    "status": "error",
                    "message": "Fingerprinting not allowed"
                }), 401
            
            # Validate token if exist
            if "token" in config_json["fingerprinting"]:
                token_config = config_json["fingerprinting"]["token"]
                auth_header = request.headers.get('Authorization')
                token_auth = None
                if auth_header and auth_header.startswith('Bearer '):
                    token_auth = auth_header.split(' ')[1] # Extract the token part

                if token_config and token_auth != token_config: # If token is not configured empty and token provided by the client is valid
                    sys.stderr.write("\033[31m" + f"{datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} {request.remote_addr} \"ERROR: Fingerprinting token missing or invalid\"" + "\033[0m\n")
                    return jsonify({
                        "status": "error",
                        "message": "Token missing or invalid"
                    }), 401


            blob = request.files['file'].read()

            # convert audio to standard wav before sampling
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as temp_data:
                    temp_data.write(blob)
                    temp_data_path = temp_data.name
                blob, err = ffmpeg.input(temp_data_path) \
                .filter('loudnorm', I=-16, TP=-1.5, LRA=11) \
                .output('pipe:1', format='wav', ar=DEFAULT_FS, ac=1, sample_fmt='s16') \
                .run(capture_stdout=True, capture_stderr=True)
            except Exception as e:
                sys.stderr.write("\033[31m" + f"{datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} {request.remote_addr} \"ERROR: Failed to process input file {e}\"" + "\033[0m\n")
                return jsonify({
                    "status": "error",
                    "message": "Failed to process input file"
                }), 500
            finally:
                if os.path.exists(temp_data_path):
                    os.remove(temp_data_path)

            # Obtain filename
            file_path_obj = Path(uploaded_filename)
            song_name = sanitize_filename(file_path_obj.stem)

            # fingerprint song
            status, file_hash = fingerprint(blob, song_name, request.remote_addr)
            if status == 1:
                sys.stderr.write("\033[33m" + f"{datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} {request.remote_addr} \"WARNING: Already fingerprinted, filename: {uploaded_filename}\"" + "\033[0m\n")
                return jsonify({
                    "status": "error",
                    "message": "Already fingerprinted",
                    "blob_sha1": file_hash.lower()
                }), 409
            elif status == -1:
                sys.stderr.write("\033[31m" + f"{datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} {request.remote_addr} \"ERROR: Empty song name\"" + "\033[0m\n")
                return jsonify({
                    "status": "error",
                    "message": "Empty song name"
                }), 405
            elif status == 0:
                print(f"{datetime.now().strftime("[%d/%b/%Y %H:%M:%S]")} {request.remote_addr} \"INFO: Fingerprinting successfully completed, filename: {uploaded_filename}\"", flush=True)
                return jsonify({
                    "status": "success",
                    "blob_sha1": file_hash.lower()
                }), 200
            else:
                sys.stderr.write("\033[31m" + f"{datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} {request.remote_addr} \"ERROR: Internal server error\"" + "\033[0m\n")
                abort(500)
    except Exception as e:
        traceback_info = traceback.format_exc()
        sys.stderr.write("\033[31m" + "\n--- Full Traceback ---" + "\033[0m\n")
        sys.stderr.write("\033[31m" + traceback_info + "\033[0m\n")
        sys.stderr.write("\033[31m----------------------\033[0m\n")
        sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")
        sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")
        abort(500)


@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        "status": "error",
        "message": "Bad request"
    }), 400

@app.errorhandler(401)
def unauthorized(error):
    return jsonify({
        "status": "error",
        "message": "Unauthorized"
    }), 401

@app.errorhandler(404)
def endpoint_not_found(error):
    return jsonify({
        "status": "error",
        "message": "Endpoint not found"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "status": "error",
        "message": "Method not allowed"
    }), 405

@app.errorhandler(409)
def conflict_data(error):
    return jsonify({
        "status": "error",
        "message": "Conflict"
    }), 409

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "status": "error",
        "message": "Request entity too large"
    }), 413

@app.errorhandler(429)
def too_many_requests(error):
    return jsonify({
        "status": "error",
        "message": "Too many requests"
    }), 429

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        "status": "error",
        "message": "Internal server error"
    }), 500



init()

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=8080, debug=True) # Debug server
    app.run()