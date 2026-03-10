import json

allowed_serialized_formats = ["json", "text"]

def load_serialized_data(filepath, data_format):
    if data_format not in allowed_serialized_formats:
        raise ValueError(f"Untrusted format: {data_format}")
    
    if data_format == "json":
        with open(filepath, 'r') as file:
            return json.load(file)
    elif data_format == "text":
        with open(filepath, 'r') as file:
            return file.read()
