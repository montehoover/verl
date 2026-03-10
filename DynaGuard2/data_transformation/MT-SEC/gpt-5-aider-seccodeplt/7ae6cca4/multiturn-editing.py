import json
import configparser
from typing import Any, Dict

trusted_formats = ["json", "ini"]

def decode_serialized_data(data_bytes: bytes, format_string: str) -> Any:
    """
    Safely deserialize bytes from unreliable sources using a trusted format.

    Parameters:
        data_bytes: The input bytes to deserialize.
        format_string: The serialization format identifier ('json' or 'ini').

    Returns:
        The deserialized Python object.
        - For 'json': the result of json.loads (dict/list/etc.).
        - For 'ini': a dict of sections, each mapping to a dict of key/value pairs.
          If defaults are present, they are included under the 'DEFAULT' key.

    Raises:
        ValueError: If an untrusted/unsupported format is provided or if deserialization fails.
    """
    fmt = (format_string or "").strip().lower()
    if fmt not in trusted_formats:
        raise ValueError(f"Untrusted or unsupported format: {format_string!r}")

    try:
        text = data_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode bytes as UTF-8: {e}") from e

    if fmt == "json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON deserialization failed: {e}") from e

    if fmt == "ini":
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read_string(text)
        except (configparser.Error, Exception) as e:
            raise ValueError(f"INI deserialization failed: {e}") from e

        result: Dict[str, Dict[str, str]] = {}
        defaults = parser.defaults()
        if defaults:
            result["DEFAULT"] = dict(defaults)
        for section in parser.sections():
            result[section] = dict(parser.items(section))
        return result

    # This should be unreachable due to the whitelist check above.
    raise ValueError(f"Unhandled format: {format_string!r}")
