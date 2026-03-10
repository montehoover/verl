import json
from typing import Any

# Whitelist of accepted formats; only these will be processed
allowed_serialized_formats = ["json", "text"]

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
    fmt = data_format.strip().lower() if isinstance(data_format, str) else None
    if fmt not in allowed_serialized_formats:
        raise ValueError(
            f"Untrusted or prohibited format: {data_format!r}. "
            f"Allowed formats: {', '.join(allowed_serialized_formats)}"
        )

    if fmt == "json":
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    if fmt == "text":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    # Defensive fallback; should be unreachable due to the whitelist check
    raise ValueError("Unsupported format")
