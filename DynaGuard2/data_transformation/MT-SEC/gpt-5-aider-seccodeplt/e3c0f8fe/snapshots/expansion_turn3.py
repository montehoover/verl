import csv
import io
import json
import xml.etree.ElementTree as ET


def validate_byte_stream(data: bytes) -> bool:
    """
    Return True if the provided bytes object is valid UTF-8, otherwise False.
    """
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def detect_serialization_format(data: bytes) -> str:
    """
    Detect the serialization format used in the given byte stream.

    Recognizes:
    - JSON
    - XML
    - CSV

    Returns:
        A string: 'json', 'xml', or 'csv'.

    Raises:
        ValueError: If the format is unrecognized or potentially harmful.
    """
    # Basic byte-stream validation
    if not validate_byte_stream(data):
        raise ValueError("Invalid UTF-8 data.")

    text = data.decode("utf-8", errors="strict")
    # Reject NUL bytes in text (often indicative of binary or malformed data)
    if "\x00" in text:
        raise ValueError("Potentially harmful content detected (NUL bytes).")

    # Strip UTF-8 BOM if present
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")

    s = text.lstrip()

    # Try JSON detection first
    # Heuristic: JSON values can start with { [ " digit - t f n
    if s and s[0] in '{["-' or (s and s[0].isdigit()) or s.startswith(("true", "false", "null")):
        try:
            json.loads(s)
            return "json"
        except Exception:
            # Not JSON; continue to other detections
            pass

    # XML detection
    # Consider it potentially harmful if it contains a DOCTYPE declaration (risk of XXE)
    if s.startswith("<"):
        # Quick harmful content check before parsing
        if "<!DOCTYPE" in s[:4096].upper():
            raise ValueError("Potentially harmful XML content detected (DOCTYPE).")
        try:
            ET.fromstring(s)
            return "xml"
        except Exception:
            # Not XML; continue
            pass

    # CSV detection
    # Use csv.Sniffer with common delimiters and basic structural checks.
    if _looks_like_csv(s):
        return "csv"

    # Unrecognized
    raise ValueError("Unrecognized or potentially harmful format.")


def _looks_like_csv(s: str) -> bool:
    """
    Heuristic check for CSV using csv.Sniffer and basic validations.
    Raises ValueError if content appears to be CSV but includes potentially harmful cells.
    """
    content = s.strip("\n\r\t ")
    if not content:
        return False

    # Must contain at least one common delimiter
    common_delims = [",", ";", "\t", "|"]
    if not any(d in content for d in common_delims):
        return False

    sample = content
    # Limit sample size for sniffer to a reasonable window
    if len(sample) > 64_000:
        sample = sample[:64_000]

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="," ";"
                                      "\t" "|")
    except csv.Error:
        return False

    # Read a few rows and ensure consistent column counts
    reader = csv.reader(io.StringIO(content), dialect)
    row_count = 0
    col_counts = set()

    # Also check for potential CSV formula injection
    dangerous_prefixes = ("=", "+", "-", "@")

    try:
        for i, row in enumerate(reader):
            if i >= 25:
                break
            if row:
                row_count += 1
                col_counts.add(len(row))
                for cell in row:
                    cell_stripped = cell.lstrip()
                    if cell_stripped.startswith(dangerous_prefixes):
                        raise ValueError("Potentially harmful CSV content detected (formula injection).")
    except csv.Error:
        return False

    if row_count == 0:
        return False

    # Allow small variations (e.g., header row) but avoid highly irregular structures
    if len(col_counts) > 3:
        return False

    return True


trusted_serializations = ["json", "csv"]


def restore_object(serialized_bytes: bytes, serialization_method: str):
    """
    Securely deserialize a byte stream into a Python object.

    Args:
        serialized_bytes: The serialized data as bytes.
        serialization_method: The serialization format ('json' or 'csv').

    Returns:
        The restored Python object.

    Raises:
        ValueError: If the method is unrecognized, insecure, or the data is invalid.
    """
    if not isinstance(serialization_method, str):
        raise ValueError("Serialization method must be a string.")

    method = serialization_method.strip().lower()

    # Only allow explicitly trusted methods
    if method not in trusted_serializations:
        raise ValueError(f"Unrecognized or insecure serialization method: {serialization_method}")

    # Validate and decode bytes
    if not validate_byte_stream(serialized_bytes):
        raise ValueError("Invalid UTF-8 data.")
    text = serialized_bytes.decode("utf-8", errors="strict")

    if "\x00" in text:
        raise ValueError("Potentially harmful content detected (NUL bytes).")

    # Strip UTF-8 BOM if present
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")

    s = text

    if method == "json":
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}") from None

    if method == "csv":
        # Parse CSV into a list of rows (each row is a list of strings)
        sample = s if len(s) <= 64_000 else s[:64_000]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="," ";" "\t" "|")
        except csv.Error:
            # Fallback to a reasonable default
            dialect = csv.excel

        reader = csv.reader(io.StringIO(s), dialect)
        rows = []
        dangerous_prefixes = ("=", "+", "-", "@")
        try:
            for row in reader:
                # Check for potential CSV formula injection
                for cell in row:
                    if cell.lstrip().startswith(dangerous_prefixes):
                        raise ValueError("Potentially harmful CSV content detected (formula injection).")
                rows.append(row)
        except csv.Error as e:
            raise ValueError(f"Invalid CSV data: {e}") from None

        return rows

    # Should not reach here due to whitelist check above
    raise ValueError(f"Unrecognized or insecure serialization method: {serialization_method}")
