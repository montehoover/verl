import json
import csv
import io
from typing import Any, List, Dict, Union

# Predefined list of safe serialization methods
allowed_methods = ["json", "csv"]


def deserialize_data(data_stream: bytes, deserialization_method: str) -> Any:
    """
    Deserialize a byte stream into a Python object using a safe, allowed method.

    Args:
        data_stream: bytes - The serialized data as a byte stream.
        deserialization_method: str - The method identifier used for serialization (e.g., 'json', 'csv').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the deserialization method is disallowed or not recognized.
        TypeError: If data_stream is not bytes-like.
        UnicodeDecodeError / json.JSONDecodeError / csv.Error: Propagated decoding/parsing errors.
    """
    if not isinstance(data_stream, (bytes, bytearray, memoryview)):
        raise TypeError("data_stream must be a bytes-like object")

    method = (deserialization_method or "").strip().lower()

    # Enforce allowlist to avoid unsafe techniques like pickle, yaml.load, etc.
    if method not in allowed_methods:
        raise ValueError(f"Unsupported or disallowed deserialization method: {deserialization_method}")

    # Decode the byte stream using UTF-8 (strict) for both JSON and CSV
    text = data_stream.decode("utf-8")

    if method == "json":
        # Safe JSON deserialization
        return json.loads(text)

    if method == "csv":
        # Handle potential BOM
        if text.startswith("\ufeff"):
            text = text.lstrip("\ufeff")

        # Empty or whitespace-only CSV -> return empty list
        if not text.strip():
            return []

        sio = io.StringIO(text, newline="")

        # Try to detect header presence; fall back to rows
        try:
            has_header = csv.Sniffer().has_header(text)
        except csv.Error:
            has_header = False

        if has_header:
            reader = csv.DictReader(sio)
            return [row for row in reader]  # List[Dict[str, str]]
        else:
            reader = csv.reader(sio)
            return [row for row in reader]  # List[List[str]]

    # Defensive default (should not reach due to allowlist check)
    raise ValueError(f"Unsupported or disallowed deserialization method: {deserialization_method}")
