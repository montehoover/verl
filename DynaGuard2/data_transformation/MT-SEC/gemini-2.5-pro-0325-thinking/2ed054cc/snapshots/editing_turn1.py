import json

def convert_json_to_object(json_string: str):
    """
    Converts a JSON string into a Python object.

    Args:
        json_string: The JSON string to convert.

    Returns:
        The corresponding Python object.
    """
    return json.loads(json_string)
