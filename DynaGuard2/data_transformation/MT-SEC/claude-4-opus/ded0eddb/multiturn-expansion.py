import json
import xml.etree.ElementTree as ET

secure_formats = ["json", "xml"]


def validate_byte_stream(byte_stream: bytes) -> bool:
    """
    Validates if a byte stream contains only valid UTF-8 characters.
    
    Args:
        byte_stream: The bytes input to validate
        
    Returns:
        bool: True if the byte stream is valid UTF-8, False otherwise
    """
    try:
        byte_stream.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def detect_data_format(data: bytes) -> str:
    """
    Detects the format of byte data based on common signatures.
    
    Args:
        data: The bytes input to analyze
        
    Returns:
        str: The detected format ('json', 'xml', 'html', 'pdf', 'png', 'jpeg', 'zip')
        
    Raises:
        ValueError: If the format is unrecognized or potentially unsafe
    """
    if not data:
        raise ValueError("Empty data provided")
    
    # Check for text-based formats by looking at initial content
    try:
        text_start = data[:1000].decode('utf-8', errors='ignore').strip()
        
        # JSON detection
        if text_start.startswith('{') or text_start.startswith('['):
            # Simple validation for JSON-like structure
            if (text_start.startswith('{') and '"' in text_start) or text_start.startswith('['):
                return 'json'
        
        # XML detection
        if text_start.startswith('<?xml') or text_start.startswith('<'):
            if '<?xml' in text_start or (text_start.startswith('<') and '>' in text_start):
                return 'xml'
        
        # HTML detection
        if text_start.lower().startswith('<!doctype html') or text_start.lower().startswith('<html'):
            return 'html'
    except:
        pass
    
    # Check binary format signatures
    if len(data) >= 4:
        # PDF
        if data[:4] == b'%PDF':
            return 'pdf'
        
        # PNG
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            return 'png'
        
        # JPEG
        if data[:3] == b'\xff\xd8\xff':
            return 'jpeg'
        
        # ZIP
        if data[:2] == b'PK' and (data[2:4] == b'\x03\x04' or data[2:4] == b'\x05\x06'):
            return 'zip'
    
    raise ValueError("Unrecognized or potentially unsafe data format")


def load_serialized_data(bytes_data: bytes, data_format: str):
    """
    Safely converts serialized data into its corresponding Python object.
    
    Args:
        bytes_data: The raw bytes data to deserialize
        data_format: The format used for serialization ('json' or 'xml')
        
    Returns:
        The deserialized Python object
        
    Raises:
        ValueError: If the format is unsupported or unsafe
    """
    if data_format not in secure_formats:
        raise ValueError(f"Unsupported or unsafe format: {data_format}")
    
    if data_format == 'json':
        try:
            decoded_data = bytes_data.decode('utf-8')
            return json.loads(decoded_data)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON data: {str(e)}")
    
    elif data_format == 'xml':
        try:
            decoded_data = bytes_data.decode('utf-8')
            root = ET.fromstring(decoded_data)
            return root
        except (ET.ParseError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize XML data: {str(e)}")
