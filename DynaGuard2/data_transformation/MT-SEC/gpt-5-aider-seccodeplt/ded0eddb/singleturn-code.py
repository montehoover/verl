import json
from typing import Any

# Predefined secure formats allowlist
secure_formats = ["json", "xml"]

try:
    # Prefer hardened XML parser if available
    from defusedxml import ElementTree as _SafeET  # type: ignore
    _HAS_DEFUSEDXML = True
except Exception:
    _HAS_DEFUSEDXML = False
    import xml.etree.ElementTree as _SafeET  # type: ignore


def load_serialized_data(bytes_data: bytes, data_format: str) -> Any:
    """
    Safely convert serialized data (bytes) into its corresponding Python object.

    Args:
        bytes_data: The serialized content as raw bytes.
        data_format: The format of the data; only 'json' and 'xml' are allowed.

    Returns:
        The deserialized Python object:
          - For 'json': standard Python types (dict, list, str, int, etc.).
          - For 'xml': an ElementTree.Element instance.

    Raises:
        TypeError: If bytes_data is not bytes, or data_format is not str.
        ValueError: If the format is unsupported/unsafe, or unsafe XML features are detected.
        json.JSONDecodeError / xml parsing errors may be raised for malformed data.
    """
    if not isinstance(bytes_data, (bytes, bytearray, memoryview)):
        raise TypeError("bytes_data must be bytes-like")
    if not isinstance(data_format, str):
        raise TypeError("data_format must be a string")

    fmt = data_format.strip().lower()
    if fmt not in secure_formats:
        raise ValueError(f"Unsupported or unsafe format: {data_format!r}")

    if fmt == "json":
        # Decode using UTF-8; json.loads also accepts str, so decode bytes first.
        # Allow bytes-like (e.g., memoryview) by converting to bytes explicitly.
        text = bytes(bytes_data).decode("utf-8")
        return json.loads(text)

    if fmt == "xml":
        # Additional defensive checks when defusedxml may not be available:
        # Reject inputs containing DOCTYPE or ENTITY declarations to mitigate XXE/expansion.
        upper_bytes = bytes(bytes_data).upper()
        if b"<!DOCTYPE" in upper_bytes or b"<!ENTITY" in upper_bytes:
            raise ValueError("Unsafe XML features detected (DOCTYPE/ENTITY not allowed)")

        # _SafeET.fromstring accepts bytes or str. Pass bytes directly.
        # If defusedxml is available, it will be used; otherwise stdlib ElementTree is used.
        return _SafeET.fromstring(bytes(bytes_data))

    # This line should be unreachable due to the allowlist check above.
    raise ValueError(f"Unsupported or unsafe format: {data_format!r}")
