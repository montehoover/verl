import json
import csv
import io
from typing import Any


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
    if not isinstance(serialized_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("serialized_bytes must be a bytes-like object")

    method = (serialization_method or "").strip().lower()

    # Rely on the externally provided trusted_serializations registry.
    try:
        allowed = set(trusted_serializations)  # type: ignore[name-defined]
    except NameError:
        # Secure-by-default fallback if registry is missing.
        allowed = {"json", "csv"}

    if method not in allowed:
        raise ValueError(f"Unrecognized or insecure serialization method: {serialization_method!r}")

    data_bytes = bytes(serialized_bytes)

    if method == "json":
        text = data_bytes.decode("utf-8-sig")
        return json.loads(text)

    if method == "csv":
        text = data_bytes.decode("utf-8-sig")
        sio = io.StringIO(text, newline="")
        reader = csv.reader(sio)
        return [row for row in reader]

    # Defensive guard: even if in allowed but not implemented.
    raise ValueError(f"Unrecognized or insecure serialization method: {serialization_method!r}")
