import json
import xml.etree.ElementTree as ET

allowed_serialization_types = ["json", "xml"]


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


def detect_serialization_format(data: bytes) -> str:
    """
    Detects the serialization format of the given byte stream.
    
    Args:
        data: The bytes input to analyze
        
    Returns:
        str: The detected format ('json', 'xml', 'yaml', 'msgpack', 'protobuf', 'pickle')
        
    Raises:
        ValueError: If the format is unrecognized or potentially unsafe
    """
    if not data:
        raise ValueError("Empty byte stream")
    
    # Check for JSON
    stripped = data.strip()
    if stripped.startswith(b'{') or stripped.startswith(b'['):
        try:
            import json
            json.loads(data.decode('utf-8'))
            return 'json'
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    
    # Check for XML
    if stripped.startswith(b'<'):
        if b'<?xml' in stripped[:50] or (b'<' in stripped and b'>' in stripped):
            return 'xml'
    
    # Check for YAML
    if data.startswith(b'---') or b': ' in data[:100] or b'- ' in data[:100]:
        try:
            if validate_byte_stream(data):
                # Basic YAML detection - could have colons or hyphens
                decoded = data.decode('utf-8')
                if '---' in decoded or ': ' in decoded or '\n- ' in decoded:
                    return 'yaml'
        except:
            pass
    
    # Check for MessagePack
    if len(data) > 0:
        first_byte = data[0]
        # MessagePack format detection based on first byte
        if (0x00 <= first_byte <= 0x7f or  # positive fixint
            0x80 <= first_byte <= 0x8f or  # fixmap
            0x90 <= first_byte <= 0x9f or  # fixarray
            0xa0 <= first_byte <= 0xbf or  # fixstr
            0xc0 <= first_byte <= 0xdf or  # various types
            0xe0 <= first_byte <= 0xff):   # negative fixint
            return 'msgpack'
    
    # Check for Protocol Buffers (basic heuristic)
    if len(data) > 2:
        # Protobuf often starts with small varint field numbers
        if data[0] in [0x08, 0x10, 0x18, 0x20, 0x28, 0x30]:
            return 'protobuf'
    
    # Check for Pickle (Python serialization) - mark as unsafe
    if data.startswith(b'\x80') and len(data) > 2 and data[1] in [0x03, 0x04, 0x05]:
        raise ValueError("Pickle format detected - potentially unsafe for untrusted data")
    
    raise ValueError("Unrecognized serialization format")


def deserialize_data(raw_data: bytes, serialization_format: str):
    """
    Safely deserializes data from raw bytes into a Python object.
    
    Args:
        raw_data: The bytes to deserialize
        serialization_format: The format used for serialization ('json' or 'xml')
        
    Returns:
        The deserialized Python object
        
    Raises:
        ValueError: If the format is unsupported or unsafe
    """
    if serialization_format not in allowed_serialization_types:
        raise ValueError(f"Unsupported serialization format: {serialization_format}")
    
    if serialization_format == 'json':
        try:
            return json.loads(raw_data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON: {e}")
    
    elif serialization_format == 'xml':
        try:
            root = ET.fromstring(raw_data.decode('utf-8'))
            return _xml_to_dict(root)
        except (ET.ParseError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize XML: {e}")


def _xml_to_dict(element):
    """Helper function to convert XML element to dictionary"""
    result = {}
    
    # Add attributes
    if element.attrib:
        result['@attributes'] = element.attrib
    
    # Add text content
    if element.text and element.text.strip():
        if len(element) == 0 and not element.attrib:
            return element.text.strip()
        else:
            result['@text'] = element.text.strip()
    
    # Add child elements
    children = {}
    for child in element:
        child_data = _xml_to_dict(child)
        if child.tag in children:
            if not isinstance(children[child.tag], list):
                children[child.tag] = [children[child.tag]]
            children[child.tag].append(child_data)
        else:
            children[child.tag] = child_data
    
    if children:
        result.update(children)
    
    # If only has text and no attributes/children, return just the text
    if len(result) == 1 and '@text' in result:
        return result['@text']
    
    return result if result else None
