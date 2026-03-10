import json
import csv
import io
from typing import Union, List, Dict

def convert_to_data_structure(data_string: str, data_format: str) -> Union[Dict, List[Dict]]:
    """
    Converts a data string (JSON or CSV) into a Python data structure.

    Args:
        data_string: The string to convert.
        data_format: The format of the data string ('json' or 'csv').

    Returns:
        A dictionary if the format is 'json'.
        A list of dictionaries if the format is 'csv'.

    Raises:
        ValueError: If the input string is not valid for the specified format,
                    or if the format is not supported.
    """
    if data_format == 'json':
        try:
            return json.loads(data_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}") from e
    elif data_format == 'csv':
        try:
            # Use io.StringIO to treat the string as a file
            csvfile = io.StringIO(data_string)
            reader = csv.DictReader(csvfile)
            return [row for row in reader]
        except csv.Error as e:
            raise ValueError(f"Invalid CSV string: {e}") from e
        except Exception as e: # Catch other potential errors during CSV processing
            raise ValueError(f"Error processing CSV string: {e}") from e
    else:
        raise ValueError(f"Unsupported data format: {data_format}. Supported formats are 'json' and 'csv'.")
