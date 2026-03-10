import json
import re
from html.parser import HTMLParser

allowed_formats = ["json", "html"]


def validate_byte_stream(byte_stream):
    """
    Validates if a byte stream contains only valid UTF-8 characters.
    
    Args:
        byte_stream (bytes): The byte stream to validate
        
    Returns:
        bool: True if the byte stream is valid UTF-8, False otherwise
    """
    try:
        # Attempt to decode the byte stream as UTF-8
        byte_stream.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def detect_data_format(byte_stream):
    """
    Detects the format of a byte stream based on common markers or headers.
    
    Args:
        byte_stream (bytes): The byte stream to analyze
        
    Returns:
        str: The detected format ('JSON', 'XML', 'HTML')
        
    Raises:
        ValueError: If the format is unrecognized or potentially harmful
    """
    # First validate that it's valid UTF-8
    if not validate_byte_stream(byte_stream):
        raise ValueError("Invalid UTF-8 byte stream")
    
    # Convert to string for analysis
    try:
        text = byte_stream.decode('utf-8').strip()
    except:
        raise ValueError("Unable to decode byte stream")
    
    if not text:
        raise ValueError("Empty byte stream")
    
    # Check for JSON
    if (text.startswith('{') and text.endswith('}')) or (text.startswith('[') and text.endswith(']')):
        # Additional validation for JSON
        try:
            json.loads(text)
            return 'JSON'
        except json.JSONDecodeError:
            pass
    
    # Check for XML
    if text.startswith('<?xml') or (text.startswith('<') and text.endswith('>')):
        # Basic XML validation
        if '<' in text and '>' in text:
            # Check for matching tags
            # Simple check for basic XML structure
            if re.search(r'<(\w+).*?>.*?</\1>', text, re.DOTALL):
                return 'XML'
    
    # Check for HTML
    html_indicators = [
        '<!DOCTYPE html',
        '<!doctype html',
        '<html',
        '<HTML',
        '<head>',
        '<HEAD>',
        '<body>',
        '<BODY>'
    ]
    
    for indicator in html_indicators:
        if indicator in text:
            return 'HTML'
    
    # If no format detected, raise ValueError
    raise ValueError("Unrecognized or potentially harmful data format")


def deserialize_content_stream(stream, stream_format):
    """
    Securely converts a byte stream into a Python object.
    
    Args:
        stream (bytes): The byte stream to deserialize
        stream_format (str): The format of the serialized data ('json' or 'html')
        
    Returns:
        object: The deserialized Python object
        
    Raises:
        ValueError: For unsupported or insecure formats
    """
    # Validate format is allowed
    if stream_format.lower() not in allowed_formats:
        raise ValueError(f"Unsupported format: {stream_format}. Allowed formats: {allowed_formats}")
    
    # Validate byte stream
    if not validate_byte_stream(stream):
        raise ValueError("Invalid UTF-8 byte stream")
    
    # Decode the stream
    try:
        content = stream.decode('utf-8')
    except Exception as e:
        raise ValueError(f"Unable to decode stream: {str(e)}")
    
    # Process based on format
    if stream_format.lower() == "json":
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
    
    elif stream_format.lower() == "html":
        # For HTML, return a simple parsed structure
        class SimpleHTMLParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.result = {
                    'tags': [],
                    'data': [],
                    'attrs': []
                }
            
            def handle_starttag(self, tag, attrs):
                self.result['tags'].append(tag)
                if attrs:
                    self.result['attrs'].extend(attrs)
            
            def handle_data(self, data):
                stripped = data.strip()
                if stripped:
                    self.result['data'].append(stripped)
        
        parser = SimpleHTMLParser()
        try:
            parser.feed(content)
            return parser.result
        except Exception as e:
            raise ValueError(f"Invalid HTML format: {str(e)}")
    
    else:
        raise ValueError(f"Format '{stream_format}' is not implemented")
