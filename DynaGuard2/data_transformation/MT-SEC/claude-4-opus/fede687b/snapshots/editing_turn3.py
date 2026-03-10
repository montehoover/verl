import json
import csv
import io

approved_formats = ["json", "csv"]

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

def deserialize_object(raw_data, format_type):
    if format_type not in approved_formats:
        raise ValueError(f"Unsupported format type: {format_type}")
    
    if format_type == 'json':
        try:
            return json.loads(raw_data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid JSON data: {e}")
    elif format_type == 'csv':
        try:
            csv_string = raw_data.decode('utf-8')
            csv_reader = csv.DictReader(io.StringIO(csv_string))
            return list(csv_reader)
        except (UnicodeDecodeError, Exception) as e:
            raise ValueError(f"Invalid CSV data: {e}")
