import json
import pickle

trusted_formats = ["json", "text"]

def read_serialized_file(file_path, file_format):
    if file_format not in trusted_formats:
        raise ValueError(f"Unsupported format: {file_format}")
    
    if file_format == "json":
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    elif file_format == "text":
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
