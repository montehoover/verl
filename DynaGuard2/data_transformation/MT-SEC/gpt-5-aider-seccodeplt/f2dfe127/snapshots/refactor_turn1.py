import json
from typing import Any

# Predefined set of trusted formats
approved_formats = ["json", "xml"]

# Try to import defusedxml for safe XML parsing
try:
    from defusedxml import ElementTree as _DefusedET  # type: ignore
    _HAS_DEFUSEDXML = True
except Exception:
    _HAS_DEFUSEDXML = False
    _DefusedET = None  # type: ignore


def bytes_to_obj(data_bytes: bytes, format_name: str) -> Any:
    """
    Deserialize raw bytes into Python objects using only approved safe formats.

    Args:
        data_bytes: The serialized object in byte format.
        format_name: The serialization format ('json', 'xml').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the format is unsupported or considered dangerous.
        TypeError: If input types are invalid.
    """
    if not isinstance(data_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("data_bytes must be a bytes-like object")
    if not isinstance(format_name, str):
        raise TypeError("format_name must be a string")

    fmt = format_name.strip().lower()
    allowed = {f.lower() for f in approved_formats}
    if fmt not in allowed:
        raise ValueError(f"Unsupported or dangerous format: {format_name!r}")

    raw_bytes = bytes(data_bytes)

    if fmt == "json":
        text = raw_bytes.decode("utf-8")
        return json.loads(text)

    if fmt == "xml":
        if not _HAS_DEFUSEDXML:
            raise ValueError("XML deserialization requires 'defusedxml' for safe parsing.")
        # Return the root Element; callers can traverse as needed.
        return _DefusedET.fromstring(raw_bytes)

    # Fallback (should not be reachable due to earlier check)
    raise ValueError(f"Unsupported or dangerous format: {format_name!r}")
