"""
Utilities for validating byte data encodings and detecting serialization formats.
"""

import json
import configparser
import xml.etree.ElementTree as ET


def validate_byte_data(data: bytes) -> bool:
    """
    Check whether the given bytes are valid UTF-8 encoded data.

    Args:
        data: Byte sequence to validate.

    Returns:
        True if the data is valid UTF-8, False otherwise.
    """
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def detect_serialization_format(data: bytes) -> str:
    """
    Determine the serialization format of the given bytes. Recognizes:
    - JSON
    - XML
    - INI

    Returns:
        A string indicating the format: "json", "xml", or "ini".

    Raises:
        TypeError: If input is not bytes-like.
        ValueError: If the format is unrecognized or potentially unsafe.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be a bytes-like object")

    b = bytes(data)

    # Reject potentially unsafe known patterns before decoding
    # Python pickle (binary protocols typically start with 0x80 <protocol> ...)
    if len(b) >= 2 and b[0] == 0x80 and 0 <= b[1] <= 9:
        raise ValueError("Potentially unsafe format detected: Python pickle data")

    # YAML with Python object tags can be unsafe under permissive loaders
    lb = b.lower()
    if b"!!python" in lb or b"tag:yaml.org,2002:python" in lb:
        raise ValueError("Potentially unsafe format detected: YAML with Python object tags")

    # Decode as UTF-8 (accept BOM if present)
    try:
        text = b.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("Unrecognized or potentially unsafe format: not valid UTF-8 text")

    s = text.strip()

    # Detect JSON
    if s and s[0] in "{[":
        try:
            json.loads(s)
            return "json"
        except Exception:
            pass  # fall through to other detectors

    # Detect XML
    if s.startswith("<"):
        # Treat presence of DOCTYPE as unsafe
        if "<!doctype" in s.lower():
            raise ValueError("Potentially unsafe XML: DOCTYPE is not allowed")
        try:
            ET.fromstring(s)
            return "xml"
        except Exception:
            pass

    # Detect INI (requires at least one section)
    cfg = configparser.ConfigParser()
    try:
        cfg.read_string(s)
        if cfg.sections():
            return "ini"
    except Exception:
        pass

    raise ValueError("Unrecognized or potentially unsafe serialization format")
