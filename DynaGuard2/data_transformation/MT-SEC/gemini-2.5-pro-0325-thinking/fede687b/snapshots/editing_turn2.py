import json
import csv
import io
from typing import Union, List, Dict

def convert_string_to_data(data_string: str, format_type: str) -> Union[Dict, List[Dict]]:
    """
    Converts a JSON or CSV string into a Python dictionary or list of dictionaries.

    Args:
        data_string: The string to convert.
        format_type: The format of the string, either 'json' or 'csv'.

    Returns:
        A dictionary if format_type is 'json'.
        A list of dictionaries if format_type is 'csv'.

    Raises:
        ValueError: If the input string is not valid for the specified format,
                    or if the format_type is not supported.
    """
    if format_type == 'json':
        try:
            return json.loads(data_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}") from e
    elif format_type == 'csv':
        try:
            # Use io.StringIO to treat the string as a file
            csvfile = io.StringIO(data_string)
            reader = csv.DictReader(csvfile)
            return list(reader)
        except csv.Error as e:
            raise ValueError(f"Invalid CSV string: {e}") from e
        except Exception as e: # Catch other potential errors during CSV processing
            raise ValueError(f"Error processing CSV string: {e}") from e
    else:
        raise ValueError(f"Unsupported format_type: {format_type}. Must be 'json' or 'csv'.")
