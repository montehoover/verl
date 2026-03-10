import json
import csv
import io
import logging
from typing import Any, List, Callable, Dict

# Predefined set of accepted serialization formats
supported_formats = ["json", "csv"]

# Module logger
logger = logging.getLogger(__name__)
_PREVIEW_MAX_BYTES = 64


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


def _input_preview_hex(data: bytes) -> str:
    """
    Return a hex-encoded preview of the input bytes (bounded length).
    """
    preview = bytes(data)[:_PREVIEW_MAX_BYTES]
    return preview.hex()


def _summarize_result(obj: Any) -> str:
    """
    Return a compact summary of the decoded result for logging purposes.
    Avoids logging full potentially sensitive payloads.
    """
    t = type(obj).__name__
    try:
        if isinstance(obj, dict):
            return f"{t}(keys={len(obj)})"
        if isinstance(obj, (list, tuple)):
            return f"{t}(len={len(obj)})"
        if isinstance(obj, (str, bytes, bytearray, memoryview)):
            return f"{t}(len={len(obj)})"
        return t
    except Exception:
        return t


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
        logger.warning("decode_data rejected input: invalid encoded_bytes type: %s", type(encoded_bytes).__name__)
        raise ValueError("encoded_bytes must be a bytes-like object")
    if not isinstance(format_name, str) or not format_name.strip():
        logger.warning("decode_data rejected input: invalid format_name: %r", format_name)
        raise ValueError("format_name must be a non-empty string")

    fmt = format_name.strip().lower()
    logger.info("Decoding data started: format=%s, bytes=%d", fmt, len(encoded_bytes))
    logger.debug("Input preview (hex, first %d bytes): %s", min(len(encoded_bytes), _PREVIEW_MAX_BYTES), _input_preview_hex(encoded_bytes))

    allowed = {f.lower() for f in supported_formats}
    if fmt not in allowed:
        logger.warning("Rejected unsupported/insecure serialization format: %r", format_name)
        raise ValueError(f"Unsupported or insecure serialization format: {format_name!r}")

    decoder = _DECODERS.get(fmt)
    if decoder is None:
        logger.error("No decoder implemented for supported format: %s", fmt)
        raise ValueError(f"Unsupported or insecure serialization format: {format_name!r}")

    try:
        result = decoder(encoded_bytes)
        logger.info("Decoding succeeded: format=%s, result_summary=%s", fmt, _summarize_result(result))
        return result
    except (UnicodeDecodeError, json.JSONDecodeError, csv.Error) as exc:
        logger.error("Decoding failed: format=%s, error=%s", fmt, exc)
        raise ValueError(f"Failed to decode data for format '{fmt}': {exc}") from exc
