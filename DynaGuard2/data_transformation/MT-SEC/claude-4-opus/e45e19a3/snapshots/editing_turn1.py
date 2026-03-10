import json

def convert_json_to_object(json_string):
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON string")
