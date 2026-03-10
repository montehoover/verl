import json
import csv
import io

def convert_json_to_dict(json_string):
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}")

def convert_string_to_data(data_string, format_type):
    if format_type == 'json':
        try:
            return json.loads(data_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")
    elif format_type == 'csv':
        try:
            csv_reader = csv.DictReader(io.StringIO(data_string))
            return list(csv_reader)
        except Exception as e:
            raise ValueError(f"Invalid CSV string: {e}")
    else:
        raise ValueError(f"Unsupported format type: {format_type}")
