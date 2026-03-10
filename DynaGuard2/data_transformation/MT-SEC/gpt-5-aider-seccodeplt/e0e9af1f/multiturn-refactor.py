import json
import logging
import os
from typing import Any, Iterable

# Whitelist of acceptable and secure serialization formats
acceptable_formats = ["json", "text"]

# Configure module-level logger to write human-readable logs to the CWD
logger = logging.getLogger("deserialization")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    log_file_path = os.path.join(os.getcwd(), "deserialization.log")
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False

def _validate_and_normalize_format(data_format: str, allowed: Iterable[str]) -> str:
    """
    Validate and normalize the provided data_format against a whitelist.

    Args:
        data_format: Format indicator provided by the caller.
        allowed: Iterable of allowed format strings.

    Returns:
        Normalized format string.

    Raises:
        ValueError: If data_format is not a string or not in the allowed list.
    """
    if not isinstance(data_format, str):
        raise ValueError("data_format must be a string")

    fmt = data_format.strip().lower()
    allowed_normalized = {str(a).strip().lower() for a in allowed}
    if fmt not in allowed_normalized:
        raise ValueError(
            f"Unsupported or unsafe format: {data_format}. "
            f"Allowed formats: {', '.join(sorted(allowed_normalized))}"
        )
    return fmt

def _deserialize_from_file(file_location: str, fmt: str) -> Any:
    """
    Deserialize the content of the file at file_location according to fmt.

    Args:
        file_location: Path to the file containing the serialized data.
        fmt: Normalized and validated format string.

    Returns:
        Deserialized Python object.

    Raises:
        ValueError: If fmt is unsupported (should not occur if validated).
        Other exceptions from file I/O or parsing may propagate.
    """
    if fmt == "json":
        logger.info("Reading file '%s' as JSON", file_location)
        with open(file_location, "r", encoding="utf-8") as f:
            return json.load(f)

    if fmt == "text":
        logger.info("Reading file '%s' as plain text", file_location)
        with open(file_location, "r", encoding="utf-8") as f:
            return f.read()

    # Redundant guard; should never be reached due to validation.
    raise ValueError(f"Unsupported or unsafe format: {fmt}")

def load_serialized_data(file_location: str, data_format: str) -> Any:
    """
    Deserialize data from a file in a secure, whitelist-restricted manner.

    Args:
        file_location: Path to the file containing the serialized data.
        data_format: Format of the serialized data. Must be one of acceptable_formats.

    Returns:
        A Python object resulting from deserializing the file contents.

    Raises:
        ValueError: If the format is unsafe or unsupported, or if inputs are not strings.
        Other exceptions (e.g., FileNotFoundError, JSONDecodeError, UnicodeDecodeError, OSError)
        may propagate from file I/O or parsing.
    """
    if not isinstance(file_location, str):
        logger.error(
            "Invalid argument: file_location must be a string, got %s",
            type(file_location).__name__
        )
        raise ValueError("file_location and data_format must be strings")

    logger.info(
        "Deserialization attempt: file='%s', format='%s'",
        file_location, data_format
    )

    try:
        fmt = _validate_and_normalize_format(data_format, acceptable_formats)
        obj = _deserialize_from_file(file_location, fmt)
        logger.info(
            "Deserialization succeeded: file='%s', format='%s', result_type='%s'",
            file_location, fmt, type(obj).__name__
        )
        return obj
    except Exception as e:
        logger.exception(
            "Deserialization failed: file='%s', format='%s' - %s",
            file_location, data_format, e
        )
        raise
