import io
import json
import csv
from typing import Any, List, Dict, Union

# Trusted serialization methods registry (provided setup)
trusted_serializations = ["json", "csv"]


def restore_object(serialized_bytes: bytes, serialization_method: str) -> Any:
    """
    Restore a Python object from a serialized data stream, only for trusted formats.

    Args:
        serialized_bytes: bytes - the data stream to deserialize.
        serialization_method: str - identifies the format used for serialization (e.g. 'json', 'csv').

    Returns:
        A Python object restored from the serialized data.

    Raises:
        ValueError: If the method is unrecognized, insecure, or the data is invalid for the given method.
    """
    if not isinstance(serialized_bytes, (bytes, bytearray)):
        raise ValueError("serialized_bytes must be of type 'bytes' or 'bytearray'")

    if not isinstance(serialization_method, str) or not serialization_method.strip():
        raise ValueError("serialization_method must be a non-empty string")

    method = serialization_method.strip().lower()

    # Validate method against trusted registry (case-insensitive)
    trusted_methods = {m.lower() for m in trusted_serializations}
    if method not in trusted_methods:
        raise ValueError(f"Unrecognized or insecure serialization method: {serialization_method}")

    # Decode bytes to text where applicable
    try:
        text = serialized_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError("Serialized data is not valid UTF-8 text") from e

    if method == "json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON data") from e

    if method == "csv":
        try:
            # Detect presence of header to decide structure
            has_header = False
            try:
                sniffer = csv.Sniffer()
                has_header = sniffer.has_header(text)
            except csv.Error:
                # If sniffing fails, fall back to assuming no header
                has_header = False

            sio = io.StringIO(text)
            if has_header:
                reader = csv.DictReader(sio)
                return list(reader)  # List[Dict[str, str]]
            else:
                reader = csv.reader(sio)
                return [list(row) for row in reader]  # List[List[str]]
        except csv.Error as e:
            raise ValueError("Invalid CSV data") from e

    # Should not be reachable due to earlier validation
    raise ValueError(f"Unrecognized or insecure serialization method: {serialization_method}")
