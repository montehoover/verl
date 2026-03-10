import json
from html.parser import HTMLParser

allowed_formats = ["json", "html"]

def deserialize_content_stream(stream, stream_format):
    if stream_format not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {stream_format}")
    
    if stream_format == "json":
        try:
            return json.loads(stream.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid JSON data: {e}")
    
    elif stream_format == "html":
        try:
            decoded_html = stream.decode('utf-8')
            parser = HTMLParser()
            parser.feed(decoded_html)
            return decoded_html
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid HTML data: {e}")
