import json
from typing import Any, Dict

def convert_json_to_dict(json_string: str) -> Dict[str, Any]:
    """
    Convert a JSON string into a Python dictionary.

    Args:
        json_string: A JSON-formatted string representing an object.

    Returns:
        A dictionary parsed from the JSON string.
    """
    return json.loads(json_string)
