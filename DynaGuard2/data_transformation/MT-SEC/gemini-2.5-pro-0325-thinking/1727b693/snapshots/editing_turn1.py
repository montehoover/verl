import json

def read_json_content(json_data: dict) -> dict:
    """
    Reads a JSON-like structure from a dictionary and returns a Python dictionary.

    The input dictionary is expected to have a 'content' key 
    whose value is a JSON string.

    Args:
        json_data: A dictionary with a 'content' key holding a JSON string.

    Returns:
        A Python dictionary parsed from the JSON string.
        Returns an empty dictionary if parsing fails or 'content' key is missing.
    """
    if 'content' not in json_data:
        return {}
        
    json_string = json_data['content']
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return {}
