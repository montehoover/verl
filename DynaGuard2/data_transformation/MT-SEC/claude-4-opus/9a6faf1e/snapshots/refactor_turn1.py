import json

trusted_formats = ["json", "text"]

def read_serialized_file(file_path: str, file_format: str):
    if file_format not in trusted_formats:
        raise ValueError(f"Unsafe or unsupported format: {file_format}")
    
    if file_format == "json":
        with open(file_path, 'r') as f:
            return json.load(f)
    elif file_format == "text":
        with open(file_path, 'r') as f:
            return f.read()
