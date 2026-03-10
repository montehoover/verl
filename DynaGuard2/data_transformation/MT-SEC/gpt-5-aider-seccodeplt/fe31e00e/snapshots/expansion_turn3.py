"""
Utilities for validating byte data encodings and detecting serialization formats.
"""

import json
import configparser
import xml.etree.ElementTree as ET

# Controlled list of formats considered safe for deserialization operations.
trusted_formats = ["json", "ini"]


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


def securely_load_data(byte_data: bytes, serialization_format: str):
    """
    Safely deserialize binary data from external sources.

    Args:
        byte_data: The raw bytes to deserialize.
        serialization_format: The serialization format used ('json', 'ini', ...).

    Returns:
        The deserialized Python object.

    Raises:
        TypeError: If inputs are of incorrect types.
        ValueError: If the format is not allowed/supported or the data is invalid.
    """
    if not isinstance(byte_data, (bytes, bytearray, memoryview)):
        raise TypeError("byte_data must be a bytes-like object")
    if not isinstance(serialization_format, str):
        raise TypeError("serialization_format must be a string")

    fmt = serialization_format.strip().lower()

    # Enforce allowlist
    if fmt not in trusted_formats:
        raise ValueError(f"Format not allowed or not supported: {serialization_format}")

    # Decode as UTF-8 text (accept BOM)
    try:
        text = bytes(byte_data).decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValueError("Input data is not valid UTF-8 text") from e

    if fmt == "json":
        # json.loads is safe for untrusted JSON text.
        return json.loads(text)

    if fmt == "ini":
        # Use a non-interpolating, strict parser to avoid unintended expansions.
        parser = configparser.ConfigParser(interpolation=None, strict=True)
        parser.read_string(text)
        # Convert to a plain dictionary of sections -> dict of key/value strings.
        result = {}
        for section in parser.sections():
            # raw=True prevents interpolation even if mistakenly enabled
            result[section] = {k: v for k, v in parser.items(section, raw=True)}
        return result

    # Should not be reached because of allowlist check, but keep for safety.
    raise ValueError(f"Unsupported serialization format: {serialization_format}")
