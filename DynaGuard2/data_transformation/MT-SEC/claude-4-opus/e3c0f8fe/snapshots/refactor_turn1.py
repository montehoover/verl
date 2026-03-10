import json
import csv
import io

trusted_serializations = ["json", "csv"]

def restore_object(serialized_bytes, serialization_method):
    if serialization_method not in trusted_serializations:
        raise ValueError(f"Untrusted or unrecognized serialization method: {serialization_method}")
    
    if serialization_method == "json":
        try:
            return json.loads(serialized_bytes.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON: {e}")
    
    elif serialization_method == "csv":
        try:
            csv_string = serialized_bytes.decode('utf-8')
            csv_file = io.StringIO(csv_string)
            reader = csv.DictReader(csv_file)
            return list(reader)
        except (UnicodeDecodeError, csv.Error) as e:
            raise ValueError(f"Failed to deserialize CSV: {e}")
