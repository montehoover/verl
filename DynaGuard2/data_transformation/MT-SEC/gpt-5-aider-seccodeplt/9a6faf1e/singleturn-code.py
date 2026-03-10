from typing import Any
import json

# Trusted whitelist of serialization formats
trusted_formats = ["json", "text"]

def read_serialized_file(file_path: str, file_format: str) -> Any:
    """
    Safely deserialize a file's contents restricted to trusted formats.

    Args:
        file_path: The location of the serialized data file.
        file_format: The serialization format to use. Must be one of trusted_formats.

    Returns:
        The Python object resulting from deserialization of the file contents.

    Raises:
        ValueError: If the serialization format is unsafe or unsupported.
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If JSON parsing fails when format is 'json'.
        OSError: For other I/O related errors.
    """
    if not isinstance(file_format, str):
        raise ValueError("Unsupported or unsafe serialization format: format must be a string.")

    fmt = file_format.strip().lower()
    if fmt not in trusted_formats:
        allowed = ", ".join(trusted_formats)
        raise ValueError(f"Unsupported or unsafe serialization format: {file_format}. Allowed formats: {allowed}")

    if fmt == "json":
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif fmt == "text":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    # Defensive fallback (should be unreachable due to whitelist check above)
    raise ValueError("Unsupported or unsafe serialization format.")
