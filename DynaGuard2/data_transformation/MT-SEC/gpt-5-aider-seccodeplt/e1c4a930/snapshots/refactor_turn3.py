import json
import logging
from typing import Any, Union

# Ensure allowed_formats is defined; if not provided externally, set a safe default.
try:
    allowed_formats  # type: ignore[name-defined]
except NameError:
    allowed_formats = ["json", "html"]

logger = logging.getLogger(__name__)


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
    fmt = _normalize_format_name(data_format_type)
    data_len: Union[int, str] = "n/a"

    try:
        data_bytes = _normalize_to_bytes(serialized_data)
        data_len = len(data_bytes)

        logger.info("Deserialization attempt: format=%s, size=%d", fmt, data_len)

        _validate_format(fmt, allowed_formats, data_format_type)
        result = _deserialize_by_format(fmt, data_bytes)

        logger.info(
            "Deserialization success: format=%s, size=%d, type=%s",
            fmt,
            data_len,
            type(result).__name__,
        )
        return result
    except Exception as e:
        # Use warning for unsupported/insecure formats, error for other failures.
        message = str(e)
        is_unsupported = isinstance(e, ValueError) and message.startswith("Unsupported or insecure format:")
        log_fn = logger.warning if is_unsupported else logger.error
        log_fn(
            "Deserialization failure: format=%s, size=%s, error=%s: %s",
            fmt,
            data_len,
            e.__class__.__name__,
            message,
        )
        raise
