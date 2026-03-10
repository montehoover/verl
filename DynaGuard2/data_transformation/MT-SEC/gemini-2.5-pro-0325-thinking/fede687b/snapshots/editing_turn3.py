import json
import csv
import io
from typing import Any, List, Dict

# Predefined set of accepted serialization formats
approved_formats = ["json", "csv"]

def deserialize_object(raw_data: bytes, format_type: str) -> Any:
    """
    Deserializes a byte stream into a Python object based on the specified format.

    Args:
        raw_data: The byte stream to deserialize.
        format_type: The format of the byte stream ('json', 'csv').

    Returns:
        The deserialized Python object.
        - For 'json', this can be a dict, list, or other JSON-compatible types.
        - For 'csv', this will be a list of dictionaries.

    Raises:
        ValueError: If the format_type is not in approved_formats,
                    or if the raw_data cannot be decoded or parsed according
                    to the specified format.
    """
    if format_type not in approved_formats:
        raise ValueError(
            f"Unsupported or insecure format_type: '{format_type}'. "
            f"Approved formats are: {', '.join(approved_formats)}."
        )

    try:
        # Decode bytes to string, assuming UTF-8 for text-based formats
        data_string = raw_data.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode raw_data as UTF-8: {e}") from e

    if format_type == 'json':
        try:
            return json.loads(data_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}") from e
    elif format_type == 'csv':
        try:
            # Use io.StringIO to treat the decoded string as a file
            csvfile = io.StringIO(data_string)
            reader = csv.DictReader(csvfile)
            return list(reader)
        except csv.Error as e:
            raise ValueError(f"Invalid CSV data: {e}") from e
        except Exception as e:  # Catch other potential errors during CSV processing
            raise ValueError(f"Error processing CSV data: {e}") from e
    
    # This part should ideally not be reached if approved_formats check is comprehensive
    # and all approved formats are handled above.
    # However, as a safeguard:
    raise ValueError(f"Internal error: Format '{format_type}' was approved but not handled.")
