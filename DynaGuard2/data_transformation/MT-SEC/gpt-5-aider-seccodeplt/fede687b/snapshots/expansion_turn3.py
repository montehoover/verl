import json
import csv
from io import StringIO
from xml.etree import ElementTree as ET

approved_formats = ["json", "csv"]


def validate_byte_data(data: bytes) -> bool:
    """
    Check whether the given bytes object contains valid UTF-8 encoded data.

    Args:
        data: A bytes object to validate.

    Returns:
        True if the data is valid UTF-8, False otherwise.
    """
    try:
        data.decode("utf-8")
        return True
    except (UnicodeDecodeError, AttributeError):
        return False


def check_serialization_format(data: bytes, format_type: str) -> bool:
    """
    Verify that the provided format_type is a recognized and safe serialization type
    and that the given bytes data conforms to that format.

    Recognized safe formats: JSON, CSV, XML.

    Returns:
        True if the format is recognized and the data appears valid for that format.

    Raises:
        ValueError: If the format is unrecognized or considered potentially insecure.
    """
    if not isinstance(data, (bytes, bytearray)):
        # Only bytes-like inputs are acceptable
        return False

    fmt = (format_type or "").strip().lower()

    # Explicitly reject known insecure formats
    insecure_formats = {
        "pickle",
        "python-pickle",
        "yaml",
        "yml",
        "php",
        "php-serialize",
        "marshal",
        "ruby-marshal",
        "java-serialization",
    }

    safe_formats = {"json", "csv", "xml"}

    if fmt in insecure_formats or fmt not in safe_formats:
        raise ValueError(f"Unrecognized or potentially insecure serialization format: {format_type!r}")

    b = bytes(data)

    if fmt == "json":
        try:
            s = b.decode("utf-8")
            json.loads(s)
            return True
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False

    if fmt == "csv":
        try:
            s = b.decode("utf-8")
        except UnicodeDecodeError:
            return False

        # Empty data is not considered valid CSV here
        if not s:
            return False

        try:
            reader = csv.reader(StringIO(s))
            # Consider valid if we can read at least one row without csv.Error
            for _ in reader:
                return True
            # No rows parsed
            return False
        except csv.Error:
            return False

    if fmt == "xml":
        # Basic hardening: reject XML with DTD/ENTITY declarations
        upper_b = b.upper()
        if b"<!DOCTYPE" in upper_b or b"<!ENTITY" in upper_b:
            raise ValueError("Potentially insecure XML content (DOCTYPE/ENTITY) is not allowed.")

        try:
            ET.fromstring(b)
            return True
        except ET.ParseError:
            return False

    # Should be unreachable due to earlier checks
    raise ValueError(f"Unrecognized or potentially insecure serialization format: {format_type!r}")


def deserialize_object(raw_data: bytes, format_type: str):
    """
    Securely deserialize input bytes into a Python object for approved formats.

    Args:
        raw_data: The serialized data as bytes.
        format_type: The serialization format (e.g., 'json', 'csv').

    Returns:
        A Python object representing the deserialized data.

    Raises:
        ValueError: If the format is unsupported/insecure or the data is invalid.
    """
    if not isinstance(raw_data, (bytes, bytearray)):
        raise ValueError("raw_data must be a bytes-like object")

    fmt = (format_type or "").strip().lower()
    if fmt not in approved_formats:
        raise ValueError(f"Unsupported or insecure serialization format: {format_type!r}")

    # Ensure UTF-8 text for text-based formats
    try:
        text = bytes(raw_data).decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError("raw_data is not valid UTF-8 for the specified format") from e

    if fmt == "json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON data") from e

    if fmt == "csv":
        try:
            reader = csv.reader(StringIO(text))
            return [row for row in reader]
        except csv.Error as e:
            raise ValueError("Invalid CSV data") from e

    # Should not be reachable due to the approved_formats check
    raise ValueError(f"Unsupported or insecure serialization format: {format_type!r}")
