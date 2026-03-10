import json
import logging
from typing import Any, Callable, Dict, List

# Whitelist of accepted, safe formats
allowed_formats: List[str] = ["json", "text"]

# Module-level logger
logger = logging.getLogger(__name__)


def _normalize_format(format_type: str) -> str:
    """
    Normalize the provided format string for comparison.
    """
    return format_type.strip().lower()


def _validate_allowed_format(normalized_format: str, allowed: List[str]) -> None:
    """
    Validate that the normalized format is in the allowed whitelist.

    Raises:
        ValueError: If the format is not allowed.
    """
    if normalized_format in allowed:
        return
    logger.warning(
        "Rejected untrusted or unsupported format: %r (allowed: %s)",
        normalized_format,
        ", ".join(allowed),
    )
    raise ValueError(
        f"Untrusted or unsupported format: {normalized_format!r}. "
        f"Allowed formats are: {', '.join(allowed)}"
    )


def _read_file_text(input_path: str) -> str:
    """
    Read and return the raw text content of a file.

    Raises:
        OSError/IOError: If the file cannot be opened or read.
    """
    logger.debug("Reading file content from '%s'", input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
    logger.debug("Read %d characters from '%s'", len(content), input_path)
    return content


def _deserialize_json(content: str) -> Any:
    """
    Deserialize JSON content into a Python object.

    Raises:
        json.JSONDecodeError: If the content is not valid JSON.
    """
    return json.loads(content)


def _deserialize_text(content: str) -> str:
    """
    Return the raw text content (identity deserialization).
    """
    return content


def _deserialize_content(normalized_format: str, content: str) -> Any:
    """
    Dispatch deserialization based on the normalized format.
    """
    deserializers: Dict[str, Callable[[str], Any]] = {
        "json": _deserialize_json,
        "text": _deserialize_text,
    }
    deserializer = deserializers.get(normalized_format)
    if deserializer is None:
        logger.warning("No deserializer available for format: %r", normalized_format)
        raise ValueError(f"Untrusted or unsupported format encountered: {normalized_format!r}")
    logger.debug("Using deserializer for format: %s", normalized_format)
    return deserializer(content)


def process_serialfile(input_path: str, format_type: str) -> Any:
    """
    Process a serialized file using secure deserialization practices.

    Args:
        input_path: Path to the serialized file.
        format_type: The format of the serialized data. Must be one of the
                     whitelisted safe formats in 'allowed_formats'.

    Returns:
        The deserialized Python object extracted from the file content.

    Raises:
        ValueError: If the provided format is not trusted or prohibited.
        OSError/IOError: Propagated if file access issues occur.
        json.JSONDecodeError: Propagated when JSON parsing fails (for json format).
    """
    if not isinstance(input_path, str) or not isinstance(format_type, str):
        logger.error("Invalid argument types: input_path=%r, format_type=%r", type(input_path), type(format_type))
        raise ValueError("input_path and format_type must be strings")

    normalized_format = _normalize_format(format_type)
    logger.info("Processing file '%s' with format '%s'", input_path, normalized_format)

    _validate_allowed_format(normalized_format, allowed_formats)

    raw_content = _read_file_text(input_path)
    result = _deserialize_content(normalized_format, raw_content)

    logger.debug("Deserialization complete; resulting type: %s", type(result).__name__)
    return result
