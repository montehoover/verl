from typing import Any
import io
import json
import csv

__all__ = ["deserialize_data"]

# Allowed deserialization methods
allowed_methods = ["json", "csv"]

def deserialize_data(data_stream: bytes, deserialization_method: str) -> Any:
    """
    Securely deserialize a byte-like data stream using only approved methods.

    Parameters:
        data_stream: Raw bytes to deserialize.
        deserialization_method: The method to use for deserialization ('json' or 'csv').

    Returns:
        The deserialized Python object.

    Raises:
        TypeError: If data_stream is not bytes-like.
        ValueError: If deserialization_method is not allowed.
        UnicodeDecodeError: If UTF-8 decoding fails.
        json.JSONDecodeError: If JSON parsing fails (when method='json').
        csv.Error: If CSV parsing encounters an error (when method='csv').
    """
    if not isinstance(data_stream, (bytes, bytearray, memoryview)):
        raise TypeError("data_stream must be a bytes-like object")

    method = deserialization_method.strip().lower()
    if method not in allowed_methods:
        raise ValueError(f"Disallowed deserialization method: {deserialization_method}")

    # Decode bytes to text using strict UTF-8 to avoid silent data corruption
    text = bytes(data_stream).decode("utf-8", errors="strict")

    if method == "json":
        return json.loads(text)

    if method == "csv":
        # Parse CSV into a list of rows (each row is a list of strings).
        with io.StringIO(text) as sio:
            reader = csv.reader(sio)
            return [row for row in reader]

    # This line should be unreachable due to the allowed_methods check.
    raise ValueError(f"Disallowed deserialization method: {deserialization_method}")
