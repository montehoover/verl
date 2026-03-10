import json
import csv
import io
from typing import Any, List, Dict, Union

# Setup provided
supported_formats = ["json", "csv"]


def decode_data(encoded_bytes: bytes, format_name: str) -> Any:
    """
    Deserialize input data from a bytes stream into a Python object.

    Args:
        encoded_bytes: bytes - the serialized input stream (UTF-8 encoded for text formats).
        format_name: str - the format used for serialization (e.g., 'json', 'csv').

    Returns:
        The deserialized Python object.
        - For 'json': a Python object resulting from json.loads (dict, list, etc.).
        - For 'csv': a list of dicts if a header row is detected; otherwise a list of lists.

    Raises:
        ValueError: If an unsupported or insecure serialization method is used.
        UnicodeDecodeError / json.JSONDecodeError / csv.Error may propagate for malformed inputs.
    """
    if not isinstance(encoded_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("encoded_bytes must be a bytes-like object")

    fmt = (format_name or "").strip().lower()
    if fmt not in supported_formats:
        raise ValueError(f"Unsupported or insecure serialization format: {format_name!r}")

    # Decode bytes to text; handle optional UTF-8 BOM.
    text = bytes(encoded_bytes).decode("utf-8-sig")

    if fmt == "json":
        # Safe JSON deserialization
        return json.loads(text)

    if fmt == "csv":
        # Attempt to detect CSV dialect and whether a header exists.
        sio = io.StringIO(text, newline="")
        sniff_text = text
        # Limit sniffing sample size to a reasonable amount to avoid high cost on very large inputs
        sample = sniff_text[:8192]

        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel

        try:
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            # Default to assuming a header exists; common for CSV inputs
            has_header = True

        sio.seek(0)
        if has_header:
            reader = csv.DictReader(sio, dialect=dialect)
            return [dict(row) for row in reader]
        else:
            reader = csv.reader(sio, dialect=dialect)
            return [row for row in reader]

    # Defensive: Should never reach here due to earlier validation
    raise ValueError(f"Unsupported or insecure serialization format: {format_name!r}")


__all__ = ["decode_data", "supported_formats"]
