import json
import csv
import io
from typing import Any

# Predefined list of safe serialization methods
allowed_methods = ["json", "csv"]


def _normalize_method(deserialization_method: str) -> str:
    if not isinstance(deserialization_method, str):
        raise ValueError("deserialization_method must be a string")
    return deserialization_method.strip().lower()


def _decode_text(data_stream: bytes) -> str:
    if not isinstance(data_stream, (bytes, bytearray, memoryview)):
        raise TypeError("data_stream must be bytes-like")
    return bytes(data_stream).decode("utf-8-sig")


def _deserialize_json_text(text: str) -> Any:
    return json.loads(text)


def _sniff_csv(sample: str):
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
    except csv.Error:
        dialect = csv.excel

    try:
        has_header = csv.Sniffer().has_header(sample)
    except csv.Error:
        has_header = False

    return dialect, has_header


def _deserialize_csv_text(text: str):
    sample = text[:1024]
    dialect, has_header = _sniff_csv(sample)

    f = io.StringIO(text)
    if has_header:
        reader = csv.DictReader(f, dialect=dialect)
        return [dict(row) for row in reader]
    else:
        reader = csv.reader(f, dialect=dialect)
        return [row for row in reader]


def deserialize_data(data_stream: bytes, deserialization_method: str) -> Any:
    """
    Deserialize a byte stream into a Python object using an authorized method.

    Args:
        data_stream: bytes representing the serialized data.
        deserialization_method: str, one of the allowed methods ('json', 'csv').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the deserialization method is disallowed or not recognized.
        UnicodeDecodeError, json.JSONDecodeError, csv.Error: for data decoding/parsing issues.
    """
    method = _normalize_method(deserialization_method)

    if method not in allowed_methods:
        raise ValueError(f"Deserialization method '{deserialization_method}' is not allowed.")

    # Decode text for text-based formats
    text = _decode_text(data_stream)

    if method == "json":
        return _deserialize_json_text(text)

    if method == "csv":
        return _deserialize_csv_text(text)

    # Should not reach here because of earlier check
    raise ValueError(f"Deserialization method '{deserialization_method}' is not recognized.")
