import json
import csv
import io
from typing import Any

supported_formats = ["json", "csv"]


def decode_data(encoded_bytes: bytes, format_name: str) -> Any:
    """
    Decode a bytes object using UTF-8 and deserialize it according to a supported format.

    :param encoded_bytes: The UTF-8 encoded bytes to decode.
    :param format_name: The format to deserialize ('json' or 'csv').
    :return: The deserialized Python object.
    :raises ValueError: If the format is unsupported or if parsing fails.
    """
    if not isinstance(format_name, str):
        raise ValueError("format_name must be a string indicating a supported format.")

    fmt = format_name.strip().lower()
    if fmt not in supported_formats:
        raise ValueError(f"Unsupported or insecure format: {format_name!r}. Supported formats: {supported_formats}")

    text = encoded_bytes.decode("utf-8").lstrip("\ufeff")  # Handle potential BOM

    if fmt == "json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError("Failed to parse JSON data.") from e

    if fmt == "csv":
        try:
            with io.StringIO(text) as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    return []
                return [dict(row) for row in reader]
        except Exception as e:
            raise ValueError("Failed to parse CSV data.") from e

    # This point should be unreachable due to the supported_formats check
    raise ValueError(f"Unsupported or insecure format: {format_name!r}.")
