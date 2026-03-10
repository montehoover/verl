import json

allowed_formats = ["json", "text"]

def process_serialfile(input_path, format_type):
    if format_type not in allowed_formats:
        raise ValueError(f"Untrusted format: {format_type}")
    
    with open(input_path, 'rb') as f:
        data = f.read()
    
    if format_type == "json":
        return json.loads(data.decode('utf-8'))
    elif format_type == "text":
        return data.decode('utf-8')
