import json
import csv
import io
from typing import Any, List, Dict

# Predefined set of accepted serialization formats
approved_formats = ["json", "csv"]


def deserialize_object(raw_data: bytes, format_type: str) -> Any:
    """
    Deserialize input bytes into a Python object, allowing only approved formats.

    Args:
        raw_data: bytes representing the serialized input stream.
        format_type: str indicating the format used for serialization (e.g., 'json', 'csv').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If an unsupported/insecure serialization method is used, or if deserialization fails.
        TypeError: If input types are incorrect.
    """
    if not isinstance(raw_data, (bytes, bytearray, memoryview)):
        raise TypeError("raw_data must be bytes-like")
    if not isinstance(format_type, str):
        raise TypeError("format_type must be a string")

    fmt = format_type.strip().lower()

    # Validate allowed/secure formats
    if fmt not in approved_formats:
        raise ValueError(f"Unsupported or insecure serialization method: {format_type}")

    try:
        if fmt == "json":
            # Strictly decode as UTF-8 and parse JSON
            text = bytes(raw_data).decode("utf-8")
            return json.loads(text)

        if fmt == "csv":
            # Decode CSV as UTF-8 (handle BOM if present) and return list of dict rows
            text = bytes(raw_data).decode("utf-8-sig")
            with io.StringIO(text, newline="") as sio:
                reader = csv.DictReader(sio)
                rows: List[Dict[str, Any]] = list(reader)
                return rows

        # This point should not be reachable due to the whitelist check above.
        raise ValueError(f"Unsupported or insecure serialization method: {format_type}")

    except (UnicodeDecodeError, json.JSONDecodeError, csv.Error) as e:
        raise ValueError(f"Failed to deserialize input as '{fmt}': {e}") from e
