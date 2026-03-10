import json
from typing import Any

# Predefined list of allowed formats deemed safe for deserialization
accepted_formats = ["json", "html"]


def decode_data_stream(data: bytes, data_format: str) -> Any:
    """
    Deserialize a byte stream into a Python object using only approved, safe formats.

    Args:
        data: bytes - The serialized stream of data.
        data_format: str - The format of the serialized data (e.g., 'json', 'html').

    Returns:
        The Python object reconstructed from the byte stream.

    Raises:
        ValueError: If the format is unsupported/unsafe or if deserialization fails.
        TypeError: If provided data is not bytes.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be of type 'bytes' or 'bytearray'")

    if not isinstance(data_format, str):
        raise TypeError("data_format must be a string")

    fmt = data_format.strip().lower()
    if fmt not in accepted_formats:
        raise ValueError(f"Unsupported or unsafe format: {data_format!r}. "
                         f"Allowed formats: {', '.join(accepted_formats)}")

    if fmt == "json":
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Failed to decode bytes as UTF-8 for JSON") from exc
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON data") from exc

    if fmt == "html":
        # For HTML, safely decode to a Unicode string and return it as-is.
        # Parsing or executing HTML is intentionally not performed here.
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Failed to decode bytes as UTF-8 for HTML") from exc

    # This line should be unreachable due to the earlier format check.
    raise ValueError(f"Unsupported or unsafe format: {data_format!r}")
