from typing import Any, Dict
import json
import configparser

# A secure set of accepted formats designated to ensure safe deserialization practices
approved_formats = ["json", "ini"]


def load_serialized_content(serialized_input: bytes, serialization_format: str) -> Any:
    """
    Deserialize binary data from potentially untrusted sources using only approved formats.

    Args:
        serialized_input: bytes - the provided serialized data in a binary form.
        serialization_format: str - the format used for deserialization (e.g., 'json', 'ini').

    Returns:
        The deserialized Python object contained within the data.

    Raises:
        TypeError: If argument types are incorrect.
        ValueError: If the format is not recognized/approved or the payload is invalid.
    """
    if not isinstance(serialized_input, (bytes, bytearray)):
        raise TypeError("serialized_input must be bytes or bytearray")
    if not isinstance(serialization_format, str):
        raise TypeError("serialization_format must be a string")

    fmt = serialization_format.strip().lower()

    if fmt not in approved_formats:
        raise ValueError(
            f"Unsupported or unsafe serialization format: {serialization_format!r}. "
            f"Allowed formats: {', '.join(approved_formats)}"
        )

    # Decode bytes as UTF-8 text (JSON and INI are text-based formats)
    try:
        text = serialized_input.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"Input is not valid UTF-8 text for format {fmt!r}: {e}") from e

    if fmt == "json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON input: {e}") from e

    if fmt == "ini":
        # Disable interpolation for straightforward, literal parsing
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read_string(text)
        except (configparser.Error, Exception) as e:
            raise ValueError(f"Invalid INI input: {e}") from e

        result: Dict[str, Dict[str, str]] = {}

        # Include defaults, if any
        defaults = dict(parser.defaults())
        if defaults:
            result["DEFAULT"] = defaults

        # Include all sections with their key/value pairs as strings
        for section in parser.sections():
            result[section] = {k: v for k, v in parser.items(section, raw=True)}

        return result

    # This line should not be reached due to earlier validation, but kept for safety.
    raise ValueError(
        f"Unsupported or unsafe serialization format: {serialization_format!r}. "
        f"Allowed formats: {', '.join(approved_formats)}"
    )
