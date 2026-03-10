import json

trusted_formats = ["json", "text"]

def validate_format(file_format: str) -> None:
    if file_format not in trusted_formats:
        raise ValueError(f"Unsafe or unsupported format: {file_format}")

def deserialize_content(file_path: str, file_format: str):
    if file_format == "json":
        with open(file_path, 'r') as f:
            return json.load(f)
    elif file_format == "text":
        with open(file_path, 'r') as f:
            return f.read()

def read_serialized_file(file_path: str, file_format: str):
    validate_format(file_format)
    return deserialize_content(file_path, file_format)
