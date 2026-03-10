import json
from typing import Any, List

# Whitelist of accepted, safe formats
allowed_formats: List[str] = ["json", "text"]


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
    if normalized_format not in allowed:
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
    with open(input_path, "r", encoding="utf-8") as f:
        return f.read()


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
    if normalized_format == "json":
        return _deserialize_json(content)
    if normalized_format == "text":
        return _deserialize_text(content)
    # Defensive programming: should never reach here due to prior validation.
    raise ValueError(f"Untrusted or unsupported format encountered: {normalized_format!r}")


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
        raise ValueError("input_path and format_type must be strings")

    normalized_format = _normalize_format(format_type)
    _validate_allowed_format(normalized_format, allowed_formats)

    raw_content = _read_file_text(input_path)
    return _deserialize_content(normalized_format, raw_content)
