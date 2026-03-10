import json
from typing import Any, Union

# Ensure allowed_formats is defined; if not provided externally, set a safe default.
try:
    allowed_formats  # type: ignore[name-defined]
except NameError:
    allowed_formats = ["json", "html"]


def _normalize_to_bytes(serialized_data: Union[bytes, bytearray, memoryview]) -> bytes:
    if isinstance(serialized_data, bytes):
        return serialized_data
    if isinstance(serialized_data, (bytearray, memoryview)):
        return bytes(serialized_data)
    raise TypeError("serialized_data must be a bytes-like object")


def _normalize_format_name(data_format_type: str) -> str:
    return (data_format_type or "").strip().lower()


def _validate_format(fmt: str, allowed: list[str], label_for_error: str) -> None:
    if fmt not in allowed:
        raise ValueError(f"Unsupported or insecure format: {label_for_error}")


def _decode_utf8(data_bytes: bytes, context: str) -> str:
    try:
        return data_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode {context} payload as UTF-8") from e


def _deserialize_json(data_bytes: bytes) -> Any:
    text = _decode_utf8(data_bytes, "JSON")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON payload") from e


def _deserialize_html(data_bytes: bytes) -> str:
    # Safely return decoded string without executing or evaluating content.
    return _decode_utf8(data_bytes, "HTML")


def _deserialize_by_format(fmt: str, data_bytes: bytes) -> Any:
    if fmt == "json":
        return _deserialize_json(data_bytes)
    if fmt == "html":
        return _deserialize_html(data_bytes)
    # Should not reach here if validation is done correctly.
    raise ValueError(f"Unsupported or insecure format: {fmt}")


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
    data_bytes = _normalize_to_bytes(serialized_data)
    fmt = _normalize_format_name(data_format_type)
    _validate_format(fmt, allowed_formats, data_format_type)
    return _deserialize_by_format(fmt, data_bytes)
