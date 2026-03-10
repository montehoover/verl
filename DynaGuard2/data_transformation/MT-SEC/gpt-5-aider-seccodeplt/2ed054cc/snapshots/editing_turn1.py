import json
from typing import Any

def convert_json_to_object(json_string: str) -> Any:
    """
    Convert a JSON string into the corresponding Python object.

    Args:
        json_string: A JSON-formatted string.

    Returns:
        The Python object represented by the JSON string.
    """
    return json.loads(json_string)
