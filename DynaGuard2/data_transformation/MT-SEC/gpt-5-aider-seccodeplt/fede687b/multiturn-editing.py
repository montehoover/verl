import json
import csv
import io
from typing import Any, Dict, List, Union

approved_formats = ["json", "csv"]

def deserialize_object(raw_data: bytes, format_type: str) -> Any:
    """
    Deserialize a bytes payload into a Python object based on the specified format.

    Args:
        raw_data: The serialized data as bytes.
        format_type: The format of the serialized data ('json', 'csv', 'pickle').

    Returns:
        - For 'json': a Python object parsed from the JSON text.
        - For 'csv': a list of dictionaries, one per CSV row.

    Raises:
        ValueError: If the format is unsupported/insecure or if deserialization fails.
    """
    fmt = format_type.strip().lower()
    allowed = {f.lower() for f in approved_formats}
    if fmt not in allowed:
        raise ValueError(f"Unsupported or insecure format: {format_type}")

    raw_bytes = bytes(raw_data)

    if fmt == "json":
        try:
            text = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as e:
            raise ValueError("Invalid JSON encoding") from e
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON") from e

    if fmt == "csv":
        try:
            text = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as e:
            raise ValueError("Invalid CSV encoding") from e
        try:
            reader = csv.DictReader(io.StringIO(text))
            return [dict(row) for row in reader]
        except csv.Error as e:
            raise ValueError("Invalid CSV") from e

    # Should not be reachable because of the approved_formats check.
    raise ValueError(f"Unsupported or insecure format: {format_type}")

def convert_string_to_data(data_string: str, format_type: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Convert a string in a specified format into structured Python data.

    Args:
        data_string: The input string containing data.
        format_type: The format of the input string. Supported values: 'json', 'csv'.

    Returns:
        - If format_type is 'json': a dictionary parsed from the JSON string.
        - If format_type is 'csv': a list of dictionaries, one per row.

    Raises:
        ValueError: If the format is unsupported, the JSON is invalid or does not
                    represent an object, or if CSV parsing fails.
    """
    fmt = format_type.strip().lower()
    if fmt == 'json':
        try:
            data = json.loads(data_string)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON") from e
        if not isinstance(data, dict):
            raise ValueError("JSON does not represent an object")
        return data
    elif fmt == 'csv':
        try:
            reader = csv.DictReader(io.StringIO(data_string))
            return [dict(row) for row in reader]
        except csv.Error as e:
            raise ValueError("Invalid CSV") from e
    else:
        raise ValueError(f"Unsupported format: {format_type}")

def convert_json_to_dict(json_string: str) -> Dict[str, Any]:
    """
    Backwards-compatible helper to convert a JSON string into a Python dictionary.

    Args:
        json_string: A string containing JSON that represents an object.

    Returns:
        A dictionary parsed from the JSON string.

    Raises:
        ValueError: If the input is not valid JSON or does not represent a JSON object.
    """
    data = convert_string_to_data(json_string, 'json')
    assert isinstance(data, dict)
    return data
