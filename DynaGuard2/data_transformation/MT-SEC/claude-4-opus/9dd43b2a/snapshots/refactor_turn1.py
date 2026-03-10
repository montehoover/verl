import json
from html.parser import HTMLParser

allowed_formats = ["json", "html"]

def parse_serialized_content(content: bytes, format_type: str):
    if format_type not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {format_type}")
    
    if format_type == "json":
        try:
            return json.loads(content.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to parse JSON: {str(e)}")
    
    elif format_type == "html":
        try:
            decoded_content = content.decode('utf-8')
            parser = HTMLParser()
            parser.feed(decoded_content)
            return decoded_content
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode HTML: {str(e)}")
