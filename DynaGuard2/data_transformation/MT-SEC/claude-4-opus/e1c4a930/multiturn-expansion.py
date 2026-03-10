import json
from html.parser import HTMLParser

allowed_formats = ["json", "html"]


def validate_byte_stream(byte_stream):
    """
    Validates if a byte stream contains only valid UTF-8 characters.
    
    Args:
        byte_stream: bytes object to validate
        
    Returns:
        bool: True if the byte stream is valid UTF-8, False otherwise
    """
    try:
        byte_stream.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def detect_data_format(byte_stream):
    """
    Detects the format of a byte stream based on common markers or headers.
    
    Args:
        byte_stream: bytes object to analyze
        
    Returns:
        str: The detected format ('JSON', 'XML', or 'HTML')
        
    Raises:
        ValueError: If the format is unrecognized or potentially unsafe
    """
    # First validate UTF-8
    if not validate_byte_stream(byte_stream):
        raise ValueError("Invalid UTF-8 byte stream")
    
    # Convert to string for analysis
    try:
        content = byte_stream.decode('utf-8').strip()
    except:
        raise ValueError("Unable to decode byte stream")
    
    if not content:
        raise ValueError("Empty byte stream")
    
    # Check for JSON
    if (content.startswith('{') and content.endswith('}')) or \
       (content.startswith('[') and content.endswith(']')):
        return 'JSON'
    
    # Check for XML
    if content.startswith('<?xml') or \
       (content.startswith('<') and content.endswith('>') and 
        not content.lower().startswith('<!doctype html') and
        not content.lower().startswith('<html')):
        return 'XML'
    
    # Check for HTML
    if content.lower().startswith('<!doctype html') or \
       content.lower().startswith('<html') or \
       (content.lower().find('<html') != -1 and 
        content.lower().find('</html>') != -1):
        return 'HTML'
    
    # Format not recognized
    raise ValueError("Unrecognized or potentially unsafe data format")


class SimpleHTMLParser(HTMLParser):
    """Simple HTML parser to extract text content"""
    def __init__(self):
        super().__init__()
        self.result = []
        self.current_tag = None
        
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        self.result.append({'tag': tag, 'attrs': dict(attrs), 'content': []})
        
    def handle_data(self, data):
        if self.result and data.strip():
            self.result[-1]['content'].append(data.strip())
            
    def handle_endtag(self, tag):
        self.current_tag = None


def deserialize_stream_payload(serialized_data, data_format_type):
    """
    Securely converts a byte stream into a Python object.
    
    Args:
        serialized_data: bytes object containing serialized data
        data_format_type: str indicating the format ('json' or 'html')
        
    Returns:
        Deserialized Python object
        
    Raises:
        ValueError: For unsupported or insecure formats
    """
    # Validate format type
    if data_format_type.lower() not in allowed_formats:
        raise ValueError(f"Unsupported format: {data_format_type}. Allowed formats: {allowed_formats}")
    
    # Validate byte stream
    if not validate_byte_stream(serialized_data):
        raise ValueError("Invalid UTF-8 byte stream")
    
    # Convert to string
    try:
        content = serialized_data.decode('utf-8')
    except Exception as e:
        raise ValueError(f"Unable to decode byte stream: {e}")
    
    # Deserialize based on format
    if data_format_type.lower() == 'json':
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
    
    elif data_format_type.lower() == 'html':
        try:
            parser = SimpleHTMLParser()
            parser.feed(content)
            return parser.result
        except Exception as e:
            raise ValueError(f"Invalid HTML data: {e}")
