import json
from typing import Any

# Only specific formats, predefined as safe, should be processed for deserialization
allowed_formats = ["json", "html"]


def deserialize_stream_payload(serialized_data: bytes, data_format_type: str) -> Any:
    """
    Deserialize a byte stream into a Python object in a secure manner.

    Args:
        serialized_data: bytes-like object containing serialized information.
        data_format_type: str indicating the format ('json' or 'html').

    Returns:
        The deserialized Python object:
          - For 'json': a Python object produced by json.loads (dict/list/etc).
          - For 'html': a decoded UTF-8 string containing the HTML content.

    Raises:
        ValueError: If the format is unsupported/insecure, if decoding fails,
                    or if the payload is invalid for the specified format.
    """
    # Normalize and validate format
    if not isinstance(data_format_type, str):
        raise ValueError("data_format_type must be a string.")
    fmt = data_format_type.strip().lower()
    if fmt not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {data_format_type!r}")

    # Ensure we have a bytes-like object
    try:
        data_bytes = bytes(serialized_data)
    except Exception as exc:
        raise ValueError("serialized_data must be a bytes-like object.") from exc

    if fmt == "json":
        # Decode as UTF-8 and parse JSON safely
        try:
            text = data_bytes.decode("utf-8")
            return json.loads(text)
        except UnicodeDecodeError as exc:
            raise ValueError("Invalid JSON payload encoding; expected UTF-8.") from exc
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON payload.") from exc

    if fmt == "html":
        # For HTML, safely decode and return as a string (no execution or parsing)
        try:
            return data_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Invalid HTML payload encoding; expected UTF-8.") from exc

    # This point should not be reachable due to earlier validation.
    raise ValueError(f"Unsupported or insecure format: {data_format_type!r}")
