import json
import configparser
from typing import Any, Dict, Union


# A secure set of accepted formats designated to ensure safe deserialization practices.
safe_formats = ["json", "ini"]


def unpack_serialized_object(byte_data: Union[bytes, bytearray, memoryview], struct_format: str) -> Any:
    """
    Deserialize binary data from potentially untrusted sources into a Python object.

    Parameters:
    - byte_data: bytes | bytearray | memoryview
        The provided serialized data in a binary form.
    - struct_format: str
        The format used for deserialization (e.g., 'json', 'ini').

    Returns:
    - Any
        The deserialized Python object contained within the data.

    Raises:
    - TypeError: If input types are invalid.
    - ValueError: If the format is not recognized or deemed unsafe.
    - UnicodeDecodeError / json.JSONDecodeError / configparser.Error: For malformed input data.
    """
    if not isinstance(byte_data, (bytes, bytearray, memoryview)):
        raise TypeError("byte_data must be bytes-like (bytes, bytearray, or memoryview)")
    if not isinstance(struct_format, str):
        raise TypeError("struct_format must be a string")

    fmt = struct_format.strip().lower()
    if fmt not in safe_formats:
        raise ValueError(f"Unsupported or unsafe format: {struct_format}")

    # Normalize to bytes then decode to text for text-based formats
    text = bytes(byte_data).decode("utf-8")

    if fmt == "json":
        return json.loads(text)

    if fmt == "ini":
        cfg = configparser.ConfigParser(interpolation=None)
        cfg.read_string(text)
        result: Dict[str, Dict[str, str]] = {}
        # Include defaults if present
        defaults = cfg.defaults()
        if defaults:
            result["DEFAULT"] = dict(defaults)
        # Include each section's key/value pairs (raw to avoid interpolation)
        for section in cfg.sections():
            result[section] = {k: v for k, v in cfg.items(section, raw=True)}
        return result

    # This point should be unreachable due to earlier validation.
    raise ValueError(f"Unsupported or unsafe format: {struct_format}")
