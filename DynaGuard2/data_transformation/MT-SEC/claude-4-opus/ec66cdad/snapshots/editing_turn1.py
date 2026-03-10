import json

def convert_json_to_dict(json_string):
    """Convert a JSON string to a Python dictionary.
    
    Args:
        json_string (str): A valid JSON string
        
    Returns:
        dict: The corresponding Python dictionary
        
    Raises:
        ValueError: If the input is not valid JSON
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
