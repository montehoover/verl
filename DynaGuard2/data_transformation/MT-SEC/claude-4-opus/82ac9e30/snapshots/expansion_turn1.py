import re
import hashlib
import struct

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
