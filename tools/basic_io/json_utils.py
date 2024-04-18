import os
import json

# basic file process

def json_read(file_path):
    try:
        with open(os.path.abspath(file_path)) as f:
            data = json.load(f)
            return data
    except:
        raise ValueError("Unable to read JSON {}".format(file_path))

def json_write(file_path, data):
    try:
        directory = os.path.dirname(os.path.abspath(file_path))
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(os.path.abspath(file_path), 'w') as f:
            json.dump(data, f)
    except:
        raise ValueError("Unable to write JSON {}".format(file_path))