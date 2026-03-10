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
