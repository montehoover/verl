import json

acceptable_formats = ["json", "text"]

def load_serialized_data(file_location, data_format):
    if data_format not in acceptable_formats:
        raise ValueError(f"Unsupported format: {data_format}")
    
    if data_format == "json":
        with open(file_location, 'r') as file:
            return json.load(file)
    elif data_format == "text":
        with open(file_location, 'r') as file:
            return file.read()
