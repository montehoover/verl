import json
from typing import Any, List

# Whitelist of accepted, safe formats
allowed_formats: List[str] = ["json", "text"]


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

    normalized_format = format_type.strip().lower()

    # Enforce whitelist strictly; anything else is prohibited
    if normalized_format not in allowed_formats:
        raise ValueError(
            f"Untrusted or unsupported format: {format_type!r}. "
            f"Allowed formats are: {', '.join(allowed_formats)}"
        )

    if normalized_format == "json":
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if normalized_format == "text":
        with open(input_path, "r", encoding="utf-8") as f:
            return f.read()

    # This is a defensive fallback; under normal circumstances this is unreachable
    raise ValueError(
        f"Untrusted or unsupported format encountered: {format_type!r}"
    )
