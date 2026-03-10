import json
import configparser
from typing import Any

# Secure set of accepted formats designated to ensure safe deserialization practices
approved_formats = ["json", "ini"]


def load_serialized_content(serialized_input: bytes, serialization_format: str) -> Any:
    """
    Deserialize binary data from potentially untrusted sources using only approved formats.

    Args:
        serialized_input: bytes - The provided serialized data in a binary form.
        serialization_format: str - The format used for deserialization (e.g., 'json', 'ini').

    Returns:
        The deserialized Python object contained within the data.

    Raises:
        TypeError: If argument types are incorrect.
        ValueError: If the format is not recognized or deemed unsafe.
        UnicodeDecodeError / json.JSONDecodeError / configparser.Error: If decoding/parsing fails.
    """
    if not isinstance(serialized_input, bytes):
        raise TypeError("serialized_input must be of type 'bytes'")
    if not isinstance(serialization_format, str):
        raise TypeError("serialization_format must be of type 'str'")

    fmt = serialization_format.strip().lower()

    # Guard clause: reject unsupported/unsafe formats immediately
    if fmt not in approved_formats:
        raise ValueError(
            f"Unsupported or unsafe serialization format: {serialization_format!r}. "
            f"Allowed formats: {', '.join(approved_formats)}"
        )

    def _load_json(data: bytes):
        text = data.decode("utf-8")
        return json.loads(text)

    def _load_ini(data: bytes):
        text = data.decode("utf-8")
        parser = configparser.ConfigParser()
        parser.read_string(text)

        result = {}
        if parser.defaults():
            result["DEFAULT"] = dict(parser.defaults())

        for section in parser.sections():
            result[section] = dict(parser.items(section))

        return result

    handlers = {
        "json": _load_json,
        "ini": _load_ini,
    }

    handler = handlers.get(fmt)
    if handler is None:
        # Defensive guard; fmt should always be in handlers if approved.
        raise ValueError(
            f"Unsupported or unsafe serialization format encountered: {serialization_format!r}"
        )

    return handler(serialized_input)
