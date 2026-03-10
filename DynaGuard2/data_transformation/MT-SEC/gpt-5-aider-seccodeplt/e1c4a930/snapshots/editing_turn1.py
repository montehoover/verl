import json
from typing import Any, Dict


def convert_json_to_dict(json_string: str) -> Dict[str, Any]:
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e.msg}") from e

    if not isinstance(data, dict):
        raise ValueError("JSON must represent an object at the top level")

    return data
