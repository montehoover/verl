import json
import csv
from io import StringIO

def convert_to_data_structure(data_string, data_format):
    """Convert a data string to a Python data structure.
    
    Args:
        data_string (str): A data formatted string
        data_format (str): Format of the data ('json' or 'csv')
        
    Returns:
        dict or list: Dictionary for JSON, list of dictionaries for CSV
        
    Raises:
        ValueError: If the format isn't supported or data is invalid
    """
    if data_format == 'json':
        try:
            return json.loads(data_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")
    elif data_format == 'csv':
        try:
            csv_file = StringIO(data_string)
            reader = csv.DictReader(csv_file)
            return list(reader)
        except Exception as e:
            raise ValueError(f"Invalid CSV string: {e}")
    else:
        raise ValueError(f"Unsupported data format: {data_format}")
