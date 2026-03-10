import io
import json
import csv
from typing import Any, List

# Predefined approved serialization formats
approved_formats = ["json", "csv"]


def deserialize_object(raw_data: bytes, format_type: str) -> Any:
    """
    Deserialize input bytes into a Python object using an approved serialization format.

    Args:
        raw_data: Serialized data as bytes.
        format_type: The serialization format ('json' or 'csv').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the format is unsupported/insecure or the data cannot be parsed.
    """
    if not isinstance(raw_data, (bytes, bytearray)):
        raise ValueError("raw_data must be bytes")

    fmt = (format_type or "").strip().lower()
    approved = {f.lower() for f in approved_formats}

    if fmt not in approved:
        raise ValueError(f"Unsupported or insecure serialization format: {format_type!r}")

    # Decode bytes to text (strip UTF-8 BOM if present)
    try:
        text = raw_data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode input as UTF-8: {e}") from e

    if fmt == "json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}") from e

    if fmt == "csv":
        try:
            with io.StringIO(text, newline="") as buf:
                reader = csv.reader(buf)
                return [row for row in reader]
        except csv.Error as e:
            raise ValueError(f"Invalid CSV data: {e}") from e

    # This should not be reached due to the earlier check.
    raise ValueError(f"Unsupported or insecure serialization format: {format_type!r}")
