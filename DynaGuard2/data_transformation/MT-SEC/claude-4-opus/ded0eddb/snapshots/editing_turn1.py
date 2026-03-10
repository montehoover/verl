import json

def convert_json_to_object(json_string):
    """
    Convert a JSON string to a Python object.
    
    Args:
        json_string (str): A JSON formatted string
        
    Returns:
        The corresponding Python object (dict, list, str, int, float, bool, or None)
        
    Raises:
        ValueError: If the input is not valid JSON
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
