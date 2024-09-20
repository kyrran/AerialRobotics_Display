import datetime
import json
import os

# ----------------------------------- JSON ------------------------------------
'''https://github.com/TommyWoodley/TommyWoodleyMEngProject'''

# Load the JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
