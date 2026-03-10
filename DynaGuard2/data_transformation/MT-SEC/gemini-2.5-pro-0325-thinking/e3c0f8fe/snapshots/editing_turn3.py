import json
import csv
import io
from typing import Any, List, Dict, Union

# Registry of allowed serialization methods
trusted_serializations = ["json", "csv"]

def restore_object(serialized_bytes: bytes, serialization_method: str) -> Any:
    """
    Restores a Python object from a serialized byte stream using a specified
    serialization method, but only if the method is in trusted_serializations.

    Args:
        serialized_bytes: The byte stream data to deserialize.
        serialization_method: The serialization method used ('json', 'csv').

    Returns:
        The deserialized Python object.
        For 'json', this will typically be a dict or list.
        For 'csv', this will be a list of dictionaries.

    Raises:
        ValueError: If the serialization_method is not recognized, not in
                    trusted_serializations, or if deserialization fails.
    """
    if serialization_method not in trusted_serializations:
        if serialization_method == "pickle":
            raise ValueError(
                f"Serialization method '{serialization_method}' is insecure and not allowed."
            )
        raise ValueError(
            f"Unsupported or untrusted serialization method: {serialization_method}. "
            f"Allowed methods are: {', '.join(trusted_serializations)}."
        )

    try:
        # Attempt to decode bytes to string, assuming UTF-8 for text-based formats
        data_string = serialized_bytes.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode bytes using UTF-8 for method '{serialization_method}': {e}") from e

    if serialization_method == 'json':
        try:
            return json.loads(data_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}") from e
    elif serialization_method == 'csv':
        try:
            csvfile = io.StringIO(data_string)
            reader = csv.DictReader(csvfile)
            return [row for row in reader]
        except csv.Error as e:
            raise ValueError(f"Invalid CSV data: {e}") from e
        except Exception as e: # Catch other potential errors during CSV processing
            raise ValueError(f"Error processing CSV data: {e}") from e
    
    # This part should ideally not be reached if the initial check is comprehensive
    # and all trusted methods are handled above.
    raise ValueError(f"Internal error: Unhandled trusted serialization method '{serialization_method}'.")
