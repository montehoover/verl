import json

def convert_json_to_dict(json_string: str) -> dict:
    """
    Converts a JSON string into a Python dictionary.

    Args:
        json_string: The JSON string to convert.

    Returns:
        The corresponding Python dictionary.
    """
    return json.loads(json_string)
