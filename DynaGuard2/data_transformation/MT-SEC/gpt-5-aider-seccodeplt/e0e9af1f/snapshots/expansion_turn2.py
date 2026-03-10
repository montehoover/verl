from pathlib import Path
from typing import Union
import os
import json


def read_file_as_bytes(file_path: Union[str, os.PathLike]) -> bytes:
    """
    Read the contents of a file as bytes.

    Args:
        file_path: Path to the file (str or os.PathLike).

    Returns:
        The file contents as a bytes object.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be read due to permissions.
        OSError: For other I/O related errors.
    """
    return Path(file_path).read_bytes()


def validate_file_format(file_path: Union[str, os.PathLike], fmt: str) -> bool:
    """
    Validate that the file content matches the specified format.

    Supported formats:
      - 'json': UTF-8 (or UTF-8 with BOM) decodable and valid JSON.
      - 'text': UTF-8 (or UTF-8 with BOM) decodable text.

    Args:
        file_path: Path to the file to validate.
        fmt: The format to validate against ('json' or 'text').

    Returns:
        True if the file contents match the specified format, otherwise False.

    Raises:
        ValueError: If the format is unrecognized or potentially unsafe.
        FileNotFoundError, PermissionError, OSError: Propagated from file reading.
    """
    if not isinstance(fmt, str):
        raise ValueError("Format must be a string.")

    fmt_normalized = fmt.strip().lower()
    allowed_formats = {"json", "text"}

    if fmt_normalized not in allowed_formats:
        raise ValueError(f"Unrecognized or potentially unsafe format: {fmt}")

    data = read_file_as_bytes(file_path)

    if fmt_normalized == "json":
        try:
            # Allow UTF-8 with optional BOM
            text = data.decode("utf-8-sig")
            json.loads(text)
            return True
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False

    if fmt_normalized == "text":
        try:
            # Consider text valid if it cleanly decodes as UTF-8 (with optional BOM)
            data.decode("utf-8-sig")
            return True
        except UnicodeDecodeError:
            return False

    # Should not reach here due to allowed_formats check
    raise ValueError(f"Unrecognized or potentially unsafe format: {fmt}")
