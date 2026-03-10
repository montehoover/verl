"""
Secure deserialization utilities.

This module provides a safe interface for deserializing text-based data
formats from a byte stream. Only explicitly allowed, text-based formats
are supported to avoid unsafe techniques (e.g., pickle).

Currently allowed methods:
- 'json'
- 'csv'
"""

import csv
import io
import json
from typing import Any, Tuple

# Predefined list of safe serialization methods
allowed_methods = ["json", "csv"]


def _normalize_method(deserialization_method: str) -> str:
    """
    Normalize and validate the deserialization method string.

    The method is stripped of surrounding whitespace and lowercased to
    enable case-insensitive comparisons.

    Args:
        deserialization_method: The method identifier provided by the caller.

    Returns:
        The normalized method name (lowercase, trimmed).

    Raises:
        ValueError: If the provided value is not a string.
    """
    if not isinstance(deserialization_method, str):
        raise ValueError("deserialization_method must be a string")
    return deserialization_method.strip().lower()


def _decode_text(data_stream: bytes) -> str:
    """
    Decode a bytes-like object to text using UTF-8 (with BOM handling).

    Args:
        data_stream: The raw byte stream to decode.

    Returns:
        The decoded string.

    Raises:
        TypeError: If the input is not bytes-like.
        UnicodeDecodeError: If UTF-8 decoding fails.
    """
    if not isinstance(data_stream, (bytes, bytearray, memoryview)):
        raise TypeError("data_stream must be bytes-like")
    return bytes(data_stream).decode("utf-8-sig")


def _deserialize_json_text(text: str) -> Any:
    """
    Deserialize a JSON string into a Python object.

    Args:
        text: A JSON-formatted string.

    Returns:
        The Python object resulting from JSON deserialization.

    Raises:
        json.JSONDecodeError: If the input is not valid JSON.
    """
    return json.loads(text)


def _sniff_csv(sample: str) -> Tuple[csv.Dialect, bool]:
    """
    Infer CSV dialect and header presence from a sample of the text.

    This uses csv.Sniffer to guess the dialect and whether the first line
    appears to be a header. If sniffing fails, a reasonable default is used.

    Args:
        sample: A sample of the CSV text (typically a prefix of the content).

    Returns:
        A tuple of (dialect, has_header) where:
            - dialect is a csv.Dialect inferred or defaulted to csv.excel.
            - has_header indicates whether a header row is likely present.
    """
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


def _deserialize_csv_text(text: str) -> list:
    """
    Deserialize a CSV string into a list of rows.

    If a header is detected, each row is represented as a dictionary
    mapping column names to values. Otherwise, each row is a list of
    field values.

    Args:
        text: A CSV-formatted string.

    Returns:
        A list of rows. Each row is either a dict (when header detected)
        or a list of strings.

    Raises:
        csv.Error: If parsing fails.
    """
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

    Only methods listed in the global 'allowed_methods' are permitted.
    This restriction avoids unsafe techniques that can lead to arbitrary
    code execution (e.g., pickle).

    Args:
        data_stream: Bytes representing the serialized data.
        deserialization_method: The method identifier ('json', 'csv').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the method is disallowed or not recognized.
        TypeError: If the data stream is not bytes-like.
        UnicodeDecodeError: If UTF-8 decoding fails.
        json.JSONDecodeError: If JSON parsing fails.
        csv.Error: If CSV parsing fails.
    """
    method = _normalize_method(deserialization_method)

    if method not in allowed_methods:
        raise ValueError(
            f"Deserialization method '{deserialization_method}' is not allowed."
        )

    # Decode text for text-based formats
    text = _decode_text(data_stream)

    if method == "json":
        return _deserialize_json_text(text)

    if method == "csv":
        return _deserialize_csv_text(text)

    # Defensive programming: this line should never be reached due to the
    # earlier method allow-list check.
    raise ValueError(
        f"Deserialization method '{deserialization_method}' is not recognized."
    )
