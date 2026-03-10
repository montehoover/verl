import json
import logging
from typing import Any

# Whitelist of trusted serialization formats
trusted_formats = ["json", "text"]

# Configure logging to a human-readable log file in the current working directory
_logger = logging.getLogger("deserialization")
if not _logger.handlers:
    _logger.setLevel(logging.INFO)
    _handler = logging.FileHandler("deserialization.log", encoding="utf-8")
    _formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    _logger.addHandler(_handler)
    _logger.propagate = False


def _normalize_and_validate_format(file_format: str) -> str:
    """
    Normalize and validate the requested serialization format.

    Args:
        file_format: The requested serialization format.

    Returns:
        The normalized (lowercased, trimmed) format string.

    Raises:
        ValueError: If the format is not a string or is unsupported/unsafe.
    """
    if not isinstance(file_format, str):
        raise ValueError("Serialization format must be a string.")

    fmt = file_format.strip().lower()
    if fmt not in trusted_formats:
        raise ValueError(f"Unsupported or unsafe serialization format: {file_format!r}")

    return fmt


def _deserialize_content(raw_content: str, fmt: str) -> Any:
    """
    Deserialize raw string content according to the specified format.

    Args:
        raw_content: The raw string content to deserialize.
        fmt: The normalized, validated format (e.g., 'json', 'text').

    Returns:
        The deserialized Python object.

    Raises:
        json.JSONDecodeError: If JSON content is invalid when fmt is 'json'.
        ValueError: If fmt is unsupported (defensive; should not happen if validated).
    """
    if fmt == "json":
        return json.loads(raw_content)
    if fmt == "text":
        return raw_content
    # Defensive fallback (should be unreachable due to validation)
    raise ValueError(f"Unsupported or unsafe serialization format: {fmt!r}")


def read_serialized_file(file_path: str, file_format: str) -> Any:
    """
    Safely deserialize the contents of a file from a trusted serialization format.

    Args:
        file_path: The location of the serialized data file.
        file_format: The serialization format. Must be one of the trusted formats.

    Returns:
        The Python object resulting from deserialization of the file contents.

    Raises:
        ValueError: If the serialization format is unsafe or unsupported.
        OSError/IOError: If there are problems reading the file.
        json.JSONDecodeError: If the file contains invalid JSON when format is 'json'.
    """
    _logger.info("Attempting deserialization | path=%s | format=%s", file_path, file_format)
    try:
        fmt = _normalize_and_validate_format(file_format)

        # Read file content using an encoding appropriate to the format
        if fmt == "json":
            with open(file_path, "r", encoding="utf-8-sig") as f:
                raw = f.read()
        elif fmt == "text":
            with open(file_path, "r", encoding="utf-8") as f:
                raw = f.read()
        else:
            # Defensive fallback (should be unreachable due to validation)
            raise ValueError(f"Unsupported or unsafe serialization format: {file_format!r}")

        result = _deserialize_content(raw, fmt)
        _logger.info("Deserialization successful | path=%s | format=%s", file_path, fmt)
        return result
    except Exception as e:
        _logger.exception("Deserialization failed | path=%s | format=%s | error=%s", file_path, file_format, str(e))
        raise
