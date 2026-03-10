import json
from typing import Any

accepted_formats = ["json", "html"]


def decode_data_stream(data: bytes, data_format: str) -> Any:
    """
    Deserialize a byte stream into a Python object for supported safe formats.

    Args:
        data: Serialized data as bytes.
        data_format: The serialization format ('json', 'html').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If data_format is unsupported or considered unsafe.
        TypeError: If data is not a bytes-like object.
        UnicodeDecodeError / json.JSONDecodeError: For invalid content in supported formats.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be a bytes-like object")

    fmt = data_format.strip().lower() if isinstance(data_format, str) else None
    if fmt not in accepted_formats:
        raise ValueError(f"Unsupported or unsafe data format: {data_format}")

    b = bytes(data)

    if fmt == "json":
        text = b.decode("utf-8")
        return json.loads(text)
    elif fmt == "html":
        return b.decode("utf-8")

    # Defensive fallback; should not be reached due to membership check above
    raise ValueError(f"Unsupported or unsafe data format: {data_format}")
