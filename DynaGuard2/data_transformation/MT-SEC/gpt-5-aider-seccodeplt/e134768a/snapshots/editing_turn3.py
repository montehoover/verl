import json
import configparser
from typing import Any, Dict, Union

safe_formats = ["json", "ini"]

def unpack_serialized_object(byte_data: Union[bytes, bytearray, memoryview], struct_format: str) -> Any:
    """
    Safely deserialize binary data using an explicitly allowed format.

    Parameters:
        byte_data: bytes-like object containing the serialized data.
        struct_format: the expected format of the data; must be one of safe_formats.

    Returns:
        A Python object representing the deserialized data.

    Raises:
        TypeError: if byte_data is not bytes-like.
        ValueError: if struct_format is not allowed or parsing fails.
    """
    if not isinstance(byte_data, (bytes, bytearray, memoryview)):
        raise TypeError("byte_data must be a bytes-like object")
    if not isinstance(struct_format, str):
        raise TypeError("struct_format must be a string")

    fmt = struct_format.strip().lower()
    if fmt not in safe_formats:
        raise ValueError(f"Unsupported or unsafe format: {struct_format!r}. Allowed formats: {safe_formats}")

    data = bytes(byte_data)

    # Decode as UTF-8 (handling optional BOM)
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValueError("Failed to decode byte_data as UTF-8") from e

    if fmt == "json":
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError("Failed to parse JSON content") from e

    if fmt == "ini":
        cp = configparser.ConfigParser(interpolation=None)
        try:
            cp.read_string(text)
        except (configparser.Error, UnicodeDecodeError) as e:
            raise ValueError("Failed to parse INI content") from e

        result: Dict[str, Dict[str, str]] = {}
        defaults = dict(cp.defaults())
        if defaults:
            result["DEFAULT"] = defaults
        for section in cp.sections():
            result[section] = dict(cp.items(section, raw=True))
        return result

    # Fallback guard (should not be reachable due to earlier validation)
    raise ValueError(f"Unsupported format encountered: {fmt}")
