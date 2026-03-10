import json

def convert_json_to_dict(json_string: str) -> dict:
    """
    Converts a JSON string into a Python dictionary.

    Args:
        json_string: The string to convert.

    Returns:
        The corresponding dictionary.

    Raises:
        ValueError: If the input string is not valid JSON.
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}") from e
