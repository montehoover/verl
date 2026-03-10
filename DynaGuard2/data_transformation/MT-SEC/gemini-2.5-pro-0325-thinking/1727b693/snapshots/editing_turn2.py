import json

def validate_content_type(json_data: dict) -> dict:
    """
    Validates the content type and reads a JSON-like structure from a dictionary.

    Checks if the input dictionary contains a 'content_type' key with the value
    'application/json' and a 'content' key whose value is a JSON string.

    Args:
        json_data: A dictionary with 'content_type' and 'content' keys.

    Returns:
        A Python dictionary parsed from the JSON string if content type is valid
        and parsing succeeds. Returns an empty dictionary otherwise.
    """
    if 'content_type' not in json_data or json_data['content_type'] != 'application/json':
        return {}
    
    if 'content' not in json_data:
        return {}
        
    json_string = json_data['content']
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return {}
