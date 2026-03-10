import json

acceptable_formats = ["json", "text"]

def validate_format(data_format: str) -> None:
    if data_format not in acceptable_formats:
        raise ValueError(f"Unsupported or unsafe format: {data_format}")

def deserialize_json(file_location: str):
    with open(file_location, 'r') as f:
        return json.load(f)

def deserialize_text(file_location: str):
    with open(file_location, 'r') as f:
        return f.read()

def load_serialized_data(file_location: str, data_format: str):
    validate_format(data_format)
    
    if data_format == "json":
        return deserialize_json(file_location)
    elif data_format == "text":
        return deserialize_text(file_location)
