import datetime
import json
import os

# -------------------------------- DIRECTORIES --------------------------------


def make_dir(prefix):
    if prefix is None:
        return None
    dir_name = get_dir_name(prefix)
    os.mkdir(f"../models/{dir_name}")
    return dir_name


def get_dir_name(prefix):
    # Get the current date and time
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M")
    dir_name = f"{prefix}_{formatted_datetime}"

    return dir_name


# ----------------------------------- JSON ------------------------------------


# Load the JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
