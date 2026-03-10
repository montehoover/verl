import json

def json_to_obj(data):
    if not isinstance(data, dict):
        raise TypeError("Input must be a dictionary")
    return len(data)
