from dejavu.core_modules.recognize_from_api import recognize_all
from dejavu.core_modules.fingerprint_from_api import fingerprint
from flask import Flask, jsonify, request, abort
import json
import secrets
import string
from datetime import datetime
from dejavu import Dejavu
from dejavu.base_classes.sorting import quick_sort_by_confidence
from dejavu.base_classes.jsonify_binary_data import jsonify_binary
from dejavu.config.settings import (CONFIG_FILE, DEFAULT_FS)
from dejavu.database_handler.result_storage import init_all_storage_db, store_result, search_result_all, if_result_token_exist_all
from pathlib import Path
from io import BytesIO
import ffmpeg
import sys
import re

config_file = CONFIG_FILE if CONFIG_FILE not in [None, '', 'config.json'] else 'config.json'

app = Flask("tunescout_api")

def init():
    try:
        print(f"TuneScout - - {datetime.now().strftime("[%d/%b/%Y %H:%M:%S]")} \"INFO: Initializing TuneScout\" -")
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
        print(f"TuneScout - - {datetime.now().strftime("[%d/%b/%Y %H:%M:%S]")} \"INFO: worker is ready to accept requests\" -")
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
def recognize_api():
    try:
        print(f"{request.remote_addr} - - {datetime.now().strftime("[%d/%b/%Y %H:%M:%S]")} \"INFO: Incoming request from client for recognition process\" -")

        if 'file' not in request.files:
            sys.stderr.write("\033[31m" + f"{request.remote_addr} - - {datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} \"ERROR: No file part in the request\" -" + "\033[0m\n")
            return jsonify({
                "error": "No file part in the request"
            }), 400

        blob = request.files['file'].read()

        # convert audio to standard wav before sampling
        convert_only = False # If the audio should be trimmed
        with open(config_file, 'r') as jsonFile:
            json_config = json.load(jsonFile)
            if "recognizing" in json_config:
                recognizing_config = json_config["recognizing"]
                if "max_duration" in recognizing_config:
                    if isinstance(recognizing_config["max_duration"], str):
                        max_duration = int(recognizing_config["max_duration"]) * 1000
                    else:
                        max_duration = recognizing_config["max_duration"] * 1000
                    if max_duration > 0:
                        try:
                            blob = ffmpeg.input('pipe:0', t=max_duration) \
                            .output('pipe:1', format='wav', ar=DEFAULT_FS, ac=2, sample_fmt='s16') \
                            .run(input=blob, capture_stdout=True, capture_stderr=True)[0]
                        except Exception as e:
                            sys.stderr.write("\033[31m" + f"{request.remote_addr} - - {datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} \"ERROR: Failed to process input file\" -" + "\033[0m\n")
                            return jsonify({
                                "error": "Failed to process input file"
                            }), 500
                    else:
                        convert_only = True
                else:
                    convert_only = True
            else:
                convert_only = True
        if convert_only:
            try:
                blob = ffmpeg.input('pipe:0') \
                .output('pipe:1', format='wav', ar=DEFAULT_FS, ac=2, sample_fmt='s16') \
                .run(input=blob, capture_stdout=True, capture_stderr=True)[0]
            except Exception as e:
                sys.stderr.write("\033[31m" + f"{request.remote_addr} - - {datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} \"ERROR: Failed to process input file\" -" + "\033[0m\n")
                return jsonify({
                    "error": "Failed to process input file"
                }), 500
        
        results = recognize_all(blob)
        results_array = [] # Here stores the results
        for result in results:
            results_array.append(jsonify_binary(result)) # Make sure that the returned data format is JSON compatible
        
        quick_sort_by_confidence(results_array) #Quick sort output results based on fingerprinted confidence
        results_array = results_array[:3] # Only return the 3 best solutions.
        results_token = generate_result_token()

        # Make sure the result token is unique
        while if_result_token_exist_all(results_token):
            results_token = generate_result_token()
            
        result_status = store_result(results_token, results_array)
        if result_status == 0:
            print(f"{request.remote_addr} - - {datetime.now().strftime("[%d/%b/%Y %H:%M:%S]")} \"INFO: Recognition result generated, token: {results_token}\" -")
            return(jsonify({"token":results_token, "results": results_array, "status": "Success"}))
        elif result_status == 1:
            return jsonify({
                "results": []
            }), 200
        else:
            sys.stderr.write("\033[31m" + f"{request.remote_addr} - - {datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} \"ERROR: Failed to store result\" -" + "\033[0m\n")
            return jsonify({
                "error": "Falied to store result"
            }), 500
    except Exception as e:
        sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")
        abort(500)

@app.route('/api/fetch/<token>', methods=['GET'])
def fetch_result_api(token):
    try:
        json_result = search_result_all(token)
        if json_result:
            print(f"{request.remote_addr} - - {datetime.now().strftime("[%d/%b/%Y %H:%M:%S]")} \"INFO: Successfully fetched result, token: {token}\" -")
            return jsonify(json_result)
        else:
            sys.stderr.write("\033[33m" + f"{request.remote_addr} - - {datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} \"WARNING: The requested data could not be found, token: {token}\" -" + "\033[0m\n")
            return jsonify({
                "error": "The requested data could not be found"
            }), 404
    except Exception as e:
        sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")
        abort(500)


@app.route('/api/fingerprint', methods=['POST'])
def fingerprint_api():
    try:
        if 'file' not in request.files:
            print(f"{request.remote_addr} - - {datetime.now().strftime("[%d/%b/%Y %H:%M:%S]")} \"INFO: Incoming request from client for fingerprinting process, filename: {None}\" -")
            sys.stderr.write("\033[31m" + f"{request.remote_addr} - - {datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} \"ERROR: No file part in the request\" -" + "\033[0m\n")
            return jsonify({
                "error": "No file part in the request"
            }), 400
        uploaded_filename = Path(request.files["file"].filename).name
        print(f"{request.remote_addr} - - {datetime.now().strftime("[%d/%b/%Y %H:%M:%S]")} \"INFO: Incoming request from client for fingerprinting process, filename: {uploaded_filename}\" -")

        with open(config_file, 'r') as jsonFile:
            config_json = json.load(jsonFile)
            allow_fingerprinting = config_json["fingerprinting"]["allow"]

            # Check if fingerprinting is allowed
            if isinstance(allow_fingerprinting, str):
                allow_fingerprinting = allow_fingerprinting.strip().lower() in ['true', '1'] # Normalize data type true, 'true', '1', 1
            
            # "allow_fingerprinting": false. Write protection enabled
            if not allow_fingerprinting:
                sys.stderr.write("\033[31m" + f"{request.remote_addr} - - {datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} \"ERROR: Fingerprinting not allowed\" -" + "\033[0m\n")
                return jsonify({
                    "error": "Fingerprinting not allowed"
                }), 401
            
            # Validate token if exist
            if "token" in config_json["fingerprinting"]:
                token_config = config_json["fingerprinting"]["token"]
                auth_header = request.headers.get('Authorization')
                token_auth = None
                if auth_header and auth_header.startswith('Bearer '):
                    token_auth = auth_header.split(' ')[1] # Extract the token part

                if token_config and token_auth != token_config: # If token is not configured empty and token provided by the client is valid
                    sys.stderr.write("\033[31m" + f"{request.remote_addr} - - {datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} \"ERROR: Fingerprinting token missing or invalid\" -" + "\033[0m\n")
                    return jsonify({
                        "error": "Token missing or invalid"
                    }), 401


            blob = request.files['file'].read()

            # convert audio to standard wav before sampling
            try:
                blob = ffmpeg.input('pipe:0') \
                .output('pipe:1', format='wav', ar=DEFAULT_FS, ac=2, sample_fmt='s16') \
                .run(input=blob, capture_stdout=True, capture_stderr=True)[0]
            except Exception as e:
                sys.stderr.write("\033[31m" + f"{request.remote_addr} - - {datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} \"ERROR: Failed to process input file\" -" + "\033[0m\n")
                return jsonify({
                    "error": "Failed to process input file"
                }), 500

            # Obtain filename
            file_path_obj = Path(uploaded_filename)
            song_name = sanitize_filename(file_path_obj.stem)

            # fingerprint song
            status, file_hash = fingerprint(blob, song_name, request.remote_addr)
            if status == 1:
                sys.stderr.write("\033[33m" + f"{request.remote_addr} - - {datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} \"WARNING: Already fingerprinted, filename: {uploaded_filename}\" -" + "\033[0m\n")
                return jsonify({
                    "status": "Already fingerprinted",
                    "blob_sha1": file_hash.lower()
                }), 409
            elif status == -1:
                sys.stderr.write("\033[31m" + f"{request.remote_addr} - - {datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} \"ERROR: Empty song name\" -" + "\033[0m\n")
                return jsonify({
                    "error": "Empty song name"
                }), 405
            elif status == 0:
                print(f"{request.remote_addr} - - {datetime.now().strftime("[%d/%b/%Y %H:%M:%S]")} \"INFO: Fingerprinting successfully completed, filename: {uploaded_filename}\" -")
                return jsonify({
                    "status": "Success",
                    "blob_sha1": file_hash.lower()
                }), 200
            else:
                sys.stderr.write("\033[31m" + f"{request.remote_addr} - - {datetime.now().strftime('[%d/%b/%Y %H:%M:%S]')} \"ERROR: Internal server error\" -" + "\033[0m\n")
                abort(500)
    except Exception as e:
        sys.stderr.write("\033[31m" + str(e) + "\033[0m\n")
        abort(500)


@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        "error": "Bad request"
    }), 400

@app.errorhandler(401)
def unauthorized(error):
    return jsonify({
        "error": "Unauthorized"
    }), 401

@app.errorhandler(404)
def endpoint_not_found(error):
    return jsonify({
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method not allowed"
    }), 405

@app.errorhandler(409)
def conflict_data(error):
    return jsonify({
        "error": "Conflict"
    }), 409

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        "error": "Internal server error"
    }), 500



init()

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=8080, debug=True) # Debug server
    app.run()