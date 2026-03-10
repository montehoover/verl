import io
import json
import csv
from typing import Any, Iterable, List

# Predefined approved serialization formats
approved_formats = ["json", "csv"]


def _normalize_and_validate_format(format_type: str, approved: Iterable[str]) -> str:
    """
    Normalize and validate the provided format against an approved list.

    Args:
        format_type: The input format string.
        approved: Iterable of approved format names.

    Returns:
        Normalized (lowercased and stripped) format string.

    Raises:
        ValueError: If the format is unsupported or insecure.
    """
    fmt = (format_type or "").strip().lower()
    approved_set = {f.lower() for f in approved}
    if fmt not in approved_set:
        raise ValueError(f"Unsupported or insecure serialization format: {format_type!r}")
    return fmt


def _decode_bytes_utf8_sig(raw_data: bytes) -> str:
    """
    Decode bytes to text using UTF-8, stripping BOM if present.

    Args:
        raw_data: Serialized data as bytes.

    Returns:
        Decoded text string.

    Raises:
        ValueError: If decoding fails.
    """
    try:
        return raw_data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode input as UTF-8: {e}") from e


def _deserialize_text(text: str, fmt: str) -> Any:
    """
    Deserialize text according to the specified format.

    Args:
        text: The input text to deserialize.
        fmt: Normalized format type ('json' or 'csv').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If deserialization fails or the format is not supported.
    """
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

    # Should not be reached if validation is correct
    raise ValueError(f"Unsupported or insecure serialization format: {fmt!r}")


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

    fmt = _normalize_and_validate_format(format_type, approved_formats)
    text = _decode_bytes_utf8_sig(raw_data)
    return _deserialize_text(text, fmt)
