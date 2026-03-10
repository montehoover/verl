import json

acceptable_formats = ["json", "text"]

def load_serialized_data(file_location: str, data_format: str):
    if data_format not in acceptable_formats:
        raise ValueError(f"Unsupported or unsafe format: {data_format}")
    
    if data_format == "json":
        with open(file_location, 'r') as f:
            return json.load(f)
    elif data_format == "text":
        with open(file_location, 'r') as f:
            return f.read()
