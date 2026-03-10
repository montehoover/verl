from pathlib import Path
from typing import Union
import json
import xml.etree.ElementTree as ET

__all__ = ["read_file_content", "validate_file_format"]


def read_file_content(file_path: Union[str, Path], encoding: str = "utf-8") -> str:
    """
    Read and return the entire content of a file as a string.

    Args:
        file_path: Path to the file (as str or pathlib.Path).
        encoding: Text encoding to use when reading the file. Defaults to UTF-8.

    Returns:
        The content of the file as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be accessed due to permissions.
        OSError: For other I/O related errors.
    """
    path = Path(file_path)
    return path.read_text(encoding=encoding)


def validate_file_format(content: str) -> str:
    """
    Determine the format of the given file content.

    Recognized formats:
      - "json": Valid JSON object/array.
      - "xml": Well-formed XML without potentially unsafe constructs (e.g., DOCTYPE/ENTITY).
      - "text": Plain text (non-binary) that is not JSON or XML.

    Raises:
        ValueError: If the content is potentially unsafe (e.g., XML with DTD/ENTITY)
                    or appears to be binary/unrecognized.
    """
    if not isinstance(content, str):
        raise ValueError("Content must be a string.")

    # Normalize leading BOM and surrounding whitespace
    s = content.lstrip("\ufeff").strip()

    # Quick binary/safety checks
    if "\x00" in s:
        raise ValueError("Content appears to be binary (contains NUL byte).")

    # Detect JSON by structure and successful parsing
    if s.startswith("{") or s.startswith("["):
        try:
            json.loads(s)
            return "json"
        except Exception:
            # Fall through to other detections
            pass

    # Detect XML heuristically and validate safely
    if s.startswith("<"):
        lowered = s.lower()
        # Reject potentially unsafe XML with DTD/ENTITY declarations
        if "<!doctype" in lowered or "<!entity" in lowered or "<![" in lowered:
            raise ValueError("Potentially unsafe XML content (DTD/ENTITY detected).")
        try:
            ET.fromstring(s)
            return "xml"
        except Exception:
            # Not well-formed XML; continue
            pass

    # Fallback: classify as plain text if it does not look like binary
    if _looks_like_text(s):
        return "text"

    raise ValueError("Unrecognized or potentially unsafe file format.")


def _looks_like_text(s: str) -> bool:
    """
    Heuristic to determine if the string looks like plain text and not binary.
    """
    if not s:
        return True
    # Count non-printable characters excluding common whitespace
    non_printable = sum(1 for ch in s if not (ch.isprintable() or ch in "\t\r\n"))
    threshold = max(5, int(len(s) * 0.3))
    return non_printable <= threshold
