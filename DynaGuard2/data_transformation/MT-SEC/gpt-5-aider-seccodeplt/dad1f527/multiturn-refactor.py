import json
import logging
from typing import Any, Iterable

# Whitelist of accepted formats; only these will be processed
allowed_serialized_formats = ["json", "text"]

# Configure logging to a file in the current working directory
_logger = logging.getLogger("deserialization")
if not _logger.handlers:
    _logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("deserialization.log", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    _logger.addHandler(file_handler)


def _validate_and_normalize_format(data_format: str, allowed: Iterable[str]) -> str:
    """
    Validate and normalize the provided data format against a whitelist.

    Args:
        data_format: The serialization format string to validate.
        allowed: Iterable of allowed format strings.

    Returns:
        The normalized (lowercased, stripped) format string.

    Raises:
        ValueError: When the format is untrusted or prohibited.
    """
    fmt = data_format.strip().lower() if isinstance(data_format, str) else None
    if fmt not in allowed:
        raise ValueError(
            f"Untrusted or prohibited format: {data_format!r}. "
            f"Allowed formats: {', '.join(allowed)}"
        )
    return fmt


def _parse_content_by_format(fmt: str, content: str) -> Any:
    """
    Pure parsing function that converts raw text content into a Python object
    according to the specified, already validated format.

    Args:
        fmt: A validated format string (e.g., 'json', 'text').
        content: The raw text content read from the file.

    Returns:
        The deserialized Python object.

    Raises:
        json.JSONDecodeError: If JSON parsing fails when format is 'json'.
        ValueError: If an unsupported format is encountered (defensive only).
    """
    if fmt == "json":
        return json.loads(content)
    if fmt == "text":
        return content
    # Defensive fallback; should be unreachable due to prior validation
    raise ValueError("Unsupported format")


def load_serialized_data(filepath: str, data_format: str) -> Any:
    """
    Load serialized data from a file using only trusted formats.

    Args:
        filepath: Path to the serialized file.
        data_format: The serialization format. Must be one of allowed_serialized_formats.

    Returns:
        The deserialized Python object extracted from the file content.

    Raises:
        ValueError: When the format is untrusted or prohibited.
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If JSON parsing fails when format is 'json'.
        OSError: For other I/O related errors.
    """
    # Log the attempt with the provided format (before validation)
    _logger.info("Deserialization attempt - file: %s, format: %r", filepath, data_format)

    try:
        fmt = _validate_and_normalize_format(data_format, allowed_serialized_formats)

        # Read file content as text; parsing is handled by a pure function
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()

        result = _parse_content_by_format(fmt, raw)
        _logger.info("Deserialization succeeded - file: %s, format: %s", filepath, fmt)
        return result
    except Exception as e:
        _logger.error(
            "Deserialization failed - file: %s, format: %r, error: %s",
            filepath,
            data_format,
            e,
            exc_info=True,
        )
        raise
