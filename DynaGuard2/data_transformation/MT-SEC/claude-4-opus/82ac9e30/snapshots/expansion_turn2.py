import re
import hashlib
import struct
import json

def is_stream_intact(data: bytes) -> bool:
    """
    Check if the incoming byte stream adheres to expected pattern.
    
    Expected pattern:
    - Starts with magic header: b'STREAM'
    - Followed by 4-byte length field (little-endian)
    - Followed by payload of specified length
    - Ends with 4-byte CRC32 checksum
    """
    
    # Minimum valid stream size: header(6) + length(4) + payload(>=1) + checksum(4)
    if len(data) < 15:
        return False
    
    # Check magic header
    if not data.startswith(b'STREAM'):
        return False
    
    # Extract length field
    try:
        payload_length = struct.unpack('<I', data[6:10])[0]
    except struct.error:
        return False
    
    # Validate total stream length
    expected_total_length = 6 + 4 + payload_length + 4
    if len(data) != expected_total_length:
        return False
    
    # Extract payload and checksum
    payload = data[10:10+payload_length]
    provided_checksum = data[10+payload_length:10+payload_length+4]
    
    # Calculate CRC32 checksum of header + length + payload
    import zlib
    calculated_checksum = struct.pack('<I', zlib.crc32(data[:10+payload_length]) & 0xffffffff)
    
    # Verify checksum
    if provided_checksum != calculated_checksum:
        return False
    
    # Additional pattern validation on payload (customizable)
    # Example: Check if payload contains only printable ASCII or specific byte patterns
    # Uncomment and modify as needed:
    # if not all(32 <= b <= 126 for b in payload):
    #     return False
    
    return True


def is_stream_intact_regex(data: bytes) -> bool:
    """
    Alternative implementation using regex pattern matching.
    
    Expected pattern:
    - Starts with 'DATA:'
    - Followed by hex-encoded content
    - Ends with '|' followed by MD5 hash
    """
    
    # Convert bytes to string for regex matching
    try:
        data_str = data.decode('ascii')
    except UnicodeDecodeError:
        return False
    
    # Define pattern: DATA:<hex_content>|<md5_hash>
    pattern = r'^DATA:([0-9a-fA-F]+)\|([0-9a-fA-F]{32})$'
    match = re.match(pattern, data_str)
    
    if not match:
        return False
    
    hex_content, provided_hash = match.groups()
    
    # Verify hex content is valid
    try:
        content_bytes = bytes.fromhex(hex_content)
    except ValueError:
        return False
    
    # Calculate MD5 hash
    calculated_hash = hashlib.md5(content_bytes).hexdigest()
    
    # Verify hash matches
    if provided_hash.lower() != calculated_hash.lower():
        return False
    
    return True


def is_stream_intact_custom(data: bytes, pattern_config: dict) -> bool:
    """
    Flexible stream validation with configurable pattern.
    
    pattern_config should contain:
    - 'header': bytes to expect at start
    - 'min_length': minimum valid stream length
    - 'max_length': maximum valid stream length (None for no limit)
    - 'allowed_bytes': set of allowed byte values (None for any)
    - 'forbidden_sequences': list of byte sequences that invalidate stream
    """
    
    # Check length constraints
    if len(data) < pattern_config.get('min_length', 0):
        return False
    
    max_length = pattern_config.get('max_length')
    if max_length is not None and len(data) > max_length:
        return False
    
    # Check header
    header = pattern_config.get('header')
    if header and not data.startswith(header):
        return False
    
    # Check allowed bytes
    allowed_bytes = pattern_config.get('allowed_bytes')
    if allowed_bytes is not None:
        if not all(b in allowed_bytes for b in data):
            return False
    
    # Check forbidden sequences
    forbidden = pattern_config.get('forbidden_sequences', [])
    for seq in forbidden:
        if seq in data:
            return False
    
    return True


def detect_serialization_format(data: bytes) -> str:
    """
    Detect the serialization format of the input bytes.
    
    Returns:
        str: The detected format ('json', 'csv', 'xml', 'msgpack', 'protobuf', 'yaml', 'pickle')
    
    Raises:
        ValueError: If format is unknown or deemed insecure
    """
    
    if not data:
        raise ValueError("Empty data stream")
    
    # Check for JSON
    try:
        # Try to decode and parse as JSON
        decoded = data.decode('utf-8')
        trimmed = decoded.strip()
        if (trimmed.startswith('{') and trimmed.endswith('}')) or \
           (trimmed.startswith('[') and trimmed.endswith(']')):
            json.loads(trimmed)
            return 'json'
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    
    # Check for CSV
    try:
        decoded = data.decode('utf-8')
        lines = decoded.strip().split('\n')
        if len(lines) > 0:
            # Check for common CSV patterns
            if ',' in lines[0] or '\t' in lines[0] or '|' in lines[0]:
                # Additional heuristics for CSV
                delimiters = [',', '\t', '|', ';']
                for delimiter in delimiters:
                    if all(delimiter in line for line in lines[:min(5, len(lines))]):
                        return 'csv'
    except UnicodeDecodeError:
        pass
    
    # Check for XML
    if data.startswith(b'<?xml') or data.startswith(b'<'):
        try:
            decoded = data.decode('utf-8')
            if re.match(r'^\s*<\?xml[^>]*\?>', decoded) or \
               re.match(r'^\s*<[a-zA-Z][\w\-\.]*(?:\s+[^>]*)?>.*</[a-zA-Z][\w\-\.]*>\s*$', decoded, re.DOTALL):
                return 'xml'
        except UnicodeDecodeError:
            pass
    
    # Check for MessagePack
    if len(data) > 0:
        first_byte = data[0]
        # MessagePack format codes
        if (0x00 <= first_byte <= 0x7f) or \
           (0x80 <= first_byte <= 0x8f) or \
           (0x90 <= first_byte <= 0x9f) or \
           (0xa0 <= first_byte <= 0xbf) or \
           first_byte in [0xc0, 0xc2, 0xc3, 0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf,
                         0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8,
                         0xd9, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf]:
            # Additional validation for msgpack
            if len(data) > 4 and data[0:2] in [b'\xde', b'\xdf', b'\xdc', b'\xdd']:
                return 'msgpack'
            elif first_byte in [0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
                               0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f]:
                return 'msgpack'
    
    # Check for Protocol Buffers (common patterns)
    if len(data) > 2:
        # Protobuf wire format uses field numbers and types
        if (data[0] & 0x07) in [0, 1, 2, 5] and data[0] >> 3 > 0:
            # Look for typical protobuf patterns
            pos = 0
            valid_protobuf = True
            while pos < min(len(data), 20):  # Check first 20 bytes
                if pos >= len(data):
                    break
                tag = data[pos]
                wire_type = tag & 0x07
                if wire_type == 0:  # Varint
                    pos += 1
                    while pos < len(data) and data[pos] & 0x80:
                        pos += 1
                    pos += 1
                elif wire_type == 1:  # 64-bit
                    pos += 9
                elif wire_type == 2:  # Length-delimited
                    pos += 1
                    length = 0
                    shift = 0
                    while pos < len(data) and data[pos] & 0x80:
                        length |= (data[pos] & 0x7f) << shift
                        shift += 7
                        pos += 1
                    if pos < len(data):
                        length |= data[pos] << shift
                        pos += 1 + length
                elif wire_type == 5:  # 32-bit
                    pos += 5
                else:
                    valid_protobuf = False
                    break
            if valid_protobuf and pos <= len(data):
                return 'protobuf'
    
    # Check for YAML
    try:
        decoded = data.decode('utf-8')
        if re.match(r'^---\s*\n', decoded) or \
           re.match(r'^[a-zA-Z_][\w]*:\s*', decoded) or \
           re.match(r'^\s*-\s+', decoded):
            return 'yaml'
    except UnicodeDecodeError:
        pass
    
    # Check for Pickle (Python serialization) - INSECURE
    if data.startswith(b'\x80\x03') or data.startswith(b'\x80\x04') or \
       data.startswith(b'\x80\x05') or data.startswith(b'(dp'):
        raise ValueError("Pickle format detected - rejected for security reasons")
    
    # Unknown format
    raise ValueError("Unknown or unsupported serialization format")
