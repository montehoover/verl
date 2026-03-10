import json
import csv
import io
from typing import Any, List, Callable, Dict

# Predefined set of accepted serialization formats
supported_formats = ["json", "csv"]


def _deserialize_json(encoded_bytes: bytes) -> Any:
    """
    Pure function to deserialize JSON bytes into a Python object.
    """
    text = bytes(encoded_bytes).decode("utf-8-sig")
    return json.loads(text)


def _deserialize_csv(encoded_bytes: bytes) -> List[List[str]]:
    """
    Pure function to deserialize CSV bytes into a list of rows (list of lists of strings).
    """
    text = bytes(encoded_bytes).decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text, newline=""))
    rows: List[List[str]] = [row for row in reader]
    return rows


_DECODERS: Dict[str, Callable[[bytes], Any]] = {
    "json": _deserialize_json,
    "csv": _deserialize_csv,
}


def decode_data(encoded_bytes: bytes, format_name: str) -> Any:
    """
    Deserialize input data into a Python object, validating permitted serialization formats.

    Args:
        encoded_bytes: Serialized input stream as bytes.
        format_name: Name of the serialization format (e.g., 'json', 'csv').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If an unsupported or insecure serialization method is used,
                    or if the data cannot be decoded/parsing fails.
    """
    if not isinstance(encoded_bytes, (bytes, bytearray, memoryview)):
        raise ValueError("encoded_bytes must be a bytes-like object")
    if not isinstance(format_name, str) or not format_name.strip():
        raise ValueError("format_name must be a non-empty string")

    fmt = format_name.strip().lower()
    allowed = {f.lower() for f in supported_formats}
    if fmt not in allowed:
        raise ValueError(f"Unsupported or insecure serialization format: {format_name!r}")

    decoder = _DECODERS.get(fmt)
    if decoder is None:
        # Even if listed in supported_formats, we require an implemented decoder.
        raise ValueError(f"Unsupported or insecure serialization format: {format_name!r}")

    try:
        return decoder(encoded_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError, csv.Error) as exc:
        raise ValueError(f"Failed to decode data for format '{fmt}': {exc}") from exc
