import json
import csv
import io
from typing import Any, List, Dict, Optional, Union


def decode_data(encoded_bytes: bytes, format_type: Optional[str] = None) -> Any:
    """
    Decode a bytes object using UTF-8 and optionally parse the resulting string.

    If format_type is 'json', attempt to parse the decoded string as JSON and
    return the parsed object (dictionary when applicable).
    If format_type is 'csv', parse the decoded string as CSV and return a list
    of dictionaries (one per row).
    If format_type is None or unrecognized, attempt JSON parsing; on failure,
    return the decoded string as-is.

    :param encoded_bytes: The UTF-8 encoded bytes to decode.
    :param format_type: Optional; 'json' or 'csv' to indicate how to interpret the decoded text.
    :return: Parsed JSON object, list of row dictionaries for CSV, or the decoded string.
    """
    text = encoded_bytes.decode("utf-8")

    ft = format_type.strip().lower() if isinstance(format_type, str) else None

    if ft == "json":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    if ft == "csv":
        try:
            with io.StringIO(text) as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    return []
                return [dict(row) for row in reader]
        except Exception:
            return text

    # Default behavior: try JSON, otherwise return raw text.
    try:
        return json.loads(text)
    except Exception:
        return text
