import json

allowed_formats = ["json", "text"]

def read_file_content(file_path):
    with open(file_path, 'rb') as file:
        return file.read()

def detect_format(data):
    # Check for JSON format
    if data.startswith(b'{') or data.startswith(b'['):
        try:
            import json
            json.loads(data.decode('utf-8'))
            return 'json'
        except:
            pass
    
    # Check for XML format
    if data.startswith(b'<?xml') or data.startswith(b'<'):
        try:
            import xml.etree.ElementTree as ET
            ET.fromstring(data.decode('utf-8'))
            return 'xml'
        except:
            pass
    
    # Check if it's plain text
    try:
        data.decode('utf-8')
        return 'plain text'
    except:
        pass
    
    raise ValueError("Format is ambiguous or unrecognizable as a trusted format")

def process_serialfile(input_path, format_type):
    if format_type not in allowed_formats:
        raise ValueError(f"Untrusted or prohibited format: {format_type}")
    
    content = read_file_content(input_path)
    
    if format_type == 'json':
        return json.loads(content.decode('utf-8'))
    elif format_type == 'text':
        return content.decode('utf-8')
    else:
        raise ValueError(f"Untrusted or prohibited format: {format_type}")
