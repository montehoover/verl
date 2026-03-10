import json
from typing import Any, Union

# Ensure allowed_formats is defined; if not provided externally, set a safe default.
try:
    allowed_formats  # type: ignore[name-defined]
except NameError:
    allowed_formats = ["json", "html"]


def deserialize_stream_payload(serialized_data: Union[bytes, bytearray, memoryview], data_format_type: str) -> Any:
    """
    Deserialize a byte stream into a Python object for approved formats only.

    Args:
        serialized_data: bytes-like object containing serialized information.
        data_format_type: str indicating the data format (e.g., 'json', 'html').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: For unsupported or insecure formats, or decoding/parsing failures.
        TypeError: If serialized_data is not bytes-like.
    """
    if not isinstance(serialized_data, (bytes, bytearray, memoryview)):
        raise TypeError("serialized_data must be a bytes-like object")

    # Normalize to immutable bytes
    if isinstance(serialized_data, (bytearray, memoryview)):
        data_bytes = bytes(serialized_data)
    else:
        data_bytes = serialized_data

    fmt = (data_format_type or "").strip().lower()
    if fmt not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {data_format_type}")

    if fmt == "json":
        try:
            text = data_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError("Failed to decode JSON payload as UTF-8") from e
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON payload") from e

    if fmt == "html":
        # For HTML, safely return a decoded string without executing or evaluating content.
        try:
            return data_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError("Failed to decode HTML payload as UTF-8") from e

    # This point should be unreachable due to the allowed_formats check above.
    raise ValueError(f"Unsupported or insecure format: {data_format_type}")
