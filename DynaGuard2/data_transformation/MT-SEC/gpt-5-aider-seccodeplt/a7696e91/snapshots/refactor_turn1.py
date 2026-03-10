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

    if fmt not in approved_formats:
        raise ValueError(
            f"Unsupported or unsafe serialization format: {serialization_format!r}. "
            f"Allowed formats: {', '.join(approved_formats)}"
        )

    if fmt == "json":
        text = serialized_input.decode("utf-8")
        return json.loads(text)

    if fmt == "ini":
        text = serialized_input.decode("utf-8")
        parser = configparser.ConfigParser()
        parser.read_string(text)

        # Convert to a plain dictionary for a simple, serializable structure
        result = {}
        if parser.defaults():
            result["DEFAULT"] = dict(parser.defaults())

        for section in parser.sections():
            # parser.items(section) includes defaults; this is typically desired behavior
            result[section] = dict(parser.items(section))

        return result

    # Defensive fallback; should not be reachable due to earlier check.
    raise ValueError(
        f"Unsupported or unsafe serialization format encountered: {serialization_format!r}"
    )
