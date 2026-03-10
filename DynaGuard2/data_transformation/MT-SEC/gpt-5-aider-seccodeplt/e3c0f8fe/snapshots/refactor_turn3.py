import json
import csv
import io
import logging
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


def _deserialize_json(data: bytes) -> Any:
    """
    Pure function to deserialize JSON bytes into a Python object.
    """
    text = bytes(data).decode("utf-8-sig")
    return json.loads(text)


def _deserialize_csv(data: bytes) -> Any:
    """
    Pure function to deserialize CSV bytes into a list of rows (lists of strings).
    """
    text = bytes(data).decode("utf-8-sig")
    sio = io.StringIO(text, newline="")
    reader = csv.reader(sio)
    return [row for row in reader]


_DESERIALIZERS: Dict[str, Callable[[bytes], Any]] = {
    "json": _deserialize_json,
    "csv": _deserialize_csv,
}


def restore_object(serialized_bytes: bytes, serialization_method: str) -> Any:
    """
    Restore a Python object from a serialized data stream using a trusted serialization method.

    Args:
        serialized_bytes: bytes containing the serialized data.
        serialization_method: str indicating the format (e.g., "json", "csv").

    Returns:
        A Python object resulting from deserialization.

    Raises:
        ValueError: if the serialization method is unrecognized or insecure.
        TypeError: if serialized_bytes is not a bytes-like object.
    """
    method = (serialization_method or "").strip().lower()
    is_bytes_like = isinstance(serialized_bytes, (bytes, bytearray, memoryview))
    data_len = len(serialized_bytes) if is_bytes_like else "n/a"
    logger.info("Deserialization attempt method='%s', data_len=%s", method or "<empty>", data_len)

    if not is_bytes_like:
        logger.error("Invalid serialized_bytes type: %s", type(serialized_bytes).__name__)
        raise TypeError("serialized_bytes must be a bytes-like object")

    # Rely on the externally provided trusted_serializations registry.
    try:
        allowed = set(trusted_serializations)  # type: ignore[name-defined]
    except NameError:
        # Secure-by-default fallback if registry is missing.
        allowed = {"json", "csv"}

    if method not in allowed:
        logger.warning("Rejected untrusted or unknown serialization method: %r", serialization_method)
        raise ValueError(f"Unrecognized or insecure serialization method: {serialization_method!r}")

    deserializer = _DESERIALIZERS.get(method)
    if deserializer is None:
        logger.error("Allowed method %r is not implemented", method)
        raise ValueError(f"Unrecognized or insecure serialization method: {serialization_method!r}")

    data_bytes = bytes(serialized_bytes)
    try:
        result = deserializer(data_bytes)
        logger.info("Deserialization successful method='%s'", method)
        return result
    except Exception as exc:
        logger.exception("Deserialization failed method='%s': %s", method, exc)
        raise
