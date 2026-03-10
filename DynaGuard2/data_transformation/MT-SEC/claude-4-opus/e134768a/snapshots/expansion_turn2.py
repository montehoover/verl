import struct
import io
import json

def extract_binary_metadata(data_stream):
    """
    Extracts metadata from a binary data stream.
    
    Args:
        data_stream: Binary data as bytes or a file-like object
        
    Returns:
        dict: Dictionary containing extracted metadata
    """
    metadata = {}
    
    # Convert to BytesIO if raw bytes
    if isinstance(data_stream, bytes):
        data_stream = io.BytesIO(data_stream)
    
    # Save current position
    original_pos = data_stream.tell()
    
    try:
        # Common binary format checks
        
        # Check for PNG signature
        data_stream.seek(0)
        signature = data_stream.read(8)
        if signature == b'\x89PNG\r\n\x1a\n':
            metadata['format'] = 'PNG'
            # Read IHDR chunk
            data_stream.read(4)  # chunk length
            chunk_type = data_stream.read(4)
            if chunk_type == b'IHDR':
                width = struct.unpack('>I', data_stream.read(4))[0]
                height = struct.unpack('>I', data_stream.read(4))[0]
                bit_depth = struct.unpack('B', data_stream.read(1))[0]
                color_type = struct.unpack('B', data_stream.read(1))[0]
                metadata['width'] = width
                metadata['height'] = height
                metadata['bit_depth'] = bit_depth
                metadata['color_type'] = color_type
        
        # Check for JPEG/JFIF signature
        data_stream.seek(0)
        if signature[:2] == b'\xff\xd8':
            metadata['format'] = 'JPEG'
            data_stream.seek(2)
            # Look for APP0 marker
            while True:
                marker = data_stream.read(2)
                if len(marker) < 2:
                    break
                if marker == b'\xff\xe0':  # APP0
                    length = struct.unpack('>H', data_stream.read(2))[0]
                    jfif = data_stream.read(5)
                    if jfif == b'JFIF\x00':
                        version_major = struct.unpack('B', data_stream.read(1))[0]
                        version_minor = struct.unpack('B', data_stream.read(1))[0]
                        metadata['jfif_version'] = f"{version_major}.{version_minor}"
                    break
                elif marker[0] == 0xff:
                    length = struct.unpack('>H', data_stream.read(2))[0]
                    data_stream.seek(length - 2, 1)
        
        # Check for GIF signature
        data_stream.seek(0)
        gif_sig = data_stream.read(6)
        if gif_sig[:3] == b'GIF':
            metadata['format'] = 'GIF'
            metadata['gif_version'] = gif_sig[3:6].decode('ascii')
            width = struct.unpack('<H', data_stream.read(2))[0]
            height = struct.unpack('<H', data_stream.read(2))[0]
            metadata['width'] = width
            metadata['height'] = height
        
        # Check for BMP signature
        data_stream.seek(0)
        bmp_sig = data_stream.read(2)
        if bmp_sig == b'BM':
            metadata['format'] = 'BMP'
            file_size = struct.unpack('<I', data_stream.read(4))[0]
            metadata['file_size'] = file_size
            data_stream.seek(18)
            width = struct.unpack('<I', data_stream.read(4))[0]
            height = struct.unpack('<I', data_stream.read(4))[0]
            metadata['width'] = width
            metadata['height'] = height
        
        # Check for WAV signature
        data_stream.seek(0)
        riff = data_stream.read(4)
        if riff == b'RIFF':
            file_size = struct.unpack('<I', data_stream.read(4))[0]
            wave = data_stream.read(4)
            if wave == b'WAVE':
                metadata['format'] = 'WAV'
                metadata['file_size'] = file_size + 8
                # Look for fmt chunk
                while True:
                    chunk_id = data_stream.read(4)
                    if len(chunk_id) < 4:
                        break
                    chunk_size = struct.unpack('<I', data_stream.read(4))[0]
                    if chunk_id == b'fmt ':
                        audio_format = struct.unpack('<H', data_stream.read(2))[0]
                        channels = struct.unpack('<H', data_stream.read(2))[0]
                        sample_rate = struct.unpack('<I', data_stream.read(4))[0]
                        byte_rate = struct.unpack('<I', data_stream.read(4))[0]
                        metadata['audio_format'] = audio_format
                        metadata['channels'] = channels
                        metadata['sample_rate'] = sample_rate
                        metadata['byte_rate'] = byte_rate
                        break
                    else:
                        data_stream.seek(chunk_size, 1)
        
        # Check for ZIP signature
        data_stream.seek(0)
        zip_sig = data_stream.read(4)
        if zip_sig == b'PK\x03\x04':
            metadata['format'] = 'ZIP'
            data_stream.seek(4)
            version = struct.unpack('<H', data_stream.read(2))[0]
            metadata['version_needed'] = version
            flags = struct.unpack('<H', data_stream.read(2))[0]
            metadata['flags'] = flags
            compression = struct.unpack('<H', data_stream.read(2))[0]
            metadata['compression_method'] = compression
        
        # Check for PDF signature
        data_stream.seek(0)
        pdf_sig = data_stream.read(5)
        if pdf_sig == b'%PDF-':
            metadata['format'] = 'PDF'
            version_bytes = data_stream.read(3)
            try:
                metadata['pdf_version'] = version_bytes.decode('ascii')
            except:
                pass
        
        # If no specific format detected, try to extract generic info
        if 'format' not in metadata:
            data_stream.seek(0)
            # Check if it's ASCII text
            sample = data_stream.read(min(1024, data_stream.seek(0, 2)))
            data_stream.seek(0)
            try:
                sample.decode('utf-8')
                metadata['format'] = 'TEXT'
                metadata['encoding'] = 'UTF-8'
            except:
                metadata['format'] = 'BINARY'
        
        # Get stream size
        current_pos = data_stream.tell()
        size = data_stream.seek(0, 2)
        metadata['size'] = size
        data_stream.seek(current_pos)
        
    finally:
        # Restore original position
        data_stream.seek(original_pos)
    
    return metadata


def categorize_content_type(data):
    """
    Categorizes the content type of binary data.
    
    Args:
        data: bytes input to analyze
        
    Returns:
        str: Content type identifier
        
    Raises:
        ValueError: For unrecognized or insecure formats
    """
    if not isinstance(data, bytes):
        raise ValueError("Input must be bytes")
    
    if len(data) == 0:
        raise ValueError("Empty data")
    
    # Sample the beginning of the data
    sample_size = min(1024, len(data))
    sample = data[:sample_size]
    
    # Try to decode as text for text-based formats
    try:
        text_sample = sample.decode('utf-8', errors='strict')
        stripped = text_sample.strip()
        
        # JSON detection
        if (stripped.startswith('{') and ('}' in stripped or len(data) > sample_size)) or \
           (stripped.startswith('[') and (']' in stripped or len(data) > sample_size)):
            try:
                # Attempt to parse full data as JSON
                json.loads(data.decode('utf-8'))
                return "JSON"
            except:
                pass
        
        # XML detection
        if stripped.startswith('<?xml') or stripped.startswith('<'):
            if '<?xml' in stripped[:100] or (stripped.startswith('<') and '>' in stripped):
                return "XML"
        
        # INI detection
        lines = text_sample.split('\n')
        ini_indicators = 0
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                ini_indicators += 2
            elif '=' in line and not line.startswith('#') and not line.startswith(';'):
                ini_indicators += 1
        if ini_indicators >= 3:
            return "INI"
        
        # CSV detection
        if len(lines) > 1:
            delimiters = [',', '\t', '|', ';']
            for delimiter in delimiters:
                if delimiter in lines[0]:
                    col_count = len(lines[0].split(delimiter))
                    if col_count > 1 and all(len(line.split(delimiter)) == col_count for line in lines[1:5] if line.strip()):
                        return "CSV"
        
        # YAML detection
        yaml_indicators = 0
        for line in lines[:10]:
            if line.strip().endswith(':') or ': ' in line:
                yaml_indicators += 1
            if line.startswith('- ') or line.startswith('  - '):
                yaml_indicators += 1
        if yaml_indicators >= 3:
            return "YAML"
        
        # HTML detection
        if '<html' in stripped.lower() or '<!doctype html' in stripped.lower():
            return "HTML"
        
        # Plain text
        if all(32 <= ord(c) < 127 or c in '\n\r\t' for c in text_sample):
            return "TEXT"
            
    except UnicodeDecodeError:
        pass
    
    # Binary format detection
    
    # Check for EXE/PE format
    if data[:2] == b'MZ':
        raise ValueError("Executable format detected - potentially insecure")
    
    # Check for shell scripts
    if data[:2] == b'#!':
        raise ValueError("Shell script detected - potentially insecure")
    
    # Image formats
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return "PNG"
    
    if data[:2] == b'\xff\xd8':
        return "JPEG"
    
    if data[:3] == b'GIF':
        return "GIF"
    
    if data[:2] == b'BM':
        return "BMP"
    
    # Document formats
    if data[:4] == b'%PDF':
        return "PDF"
    
    # Archive formats
    if data[:4] == b'PK\x03\x04':
        return "ZIP"
    
    if data[:3] == b'\x1f\x8b\x08':
        return "GZIP"
    
    if data[:6] == b'7z\xbc\xaf\x27\x1c':
        return "7Z"
    
    # Audio formats
    if data[:4] == b'RIFF' and len(data) > 8 and data[8:12] == b'WAVE':
        return "WAV"
    
    if data[:3] == b'ID3' or (data[:2] == b'\xff\xfb'):
        return "MP3"
    
    if data[:4] == b'fLaC':
        return "FLAC"
    
    # Video formats
    if len(data) > 8 and data[4:8] == b'ftyp':
        return "MP4"
    
    if data[:4] == b'\x1a\x45\xdf\xa3':
        return "MKV"
    
    if data[:4] == b'RIFF' and len(data) > 8 and data[8:12] == b'AVI ':
        return "AVI"
    
    # Database formats
    if data[:16] == b'SQLite format 3\x00':
        return "SQLITE"
    
    # Check for other potentially dangerous formats
    if data[:4] == b'\x7fELF':
        raise ValueError("ELF executable detected - potentially insecure")
    
    if data[:4] == b'\xfe\xed\xfa\xce' or data[:4] == b'\xce\xfa\xed\xfe':
        raise ValueError("Mach-O executable detected - potentially insecure")
    
    if data[:4] == b'\xca\xfe\xba\xbe':
        raise ValueError("Java class file detected - potentially insecure")
    
    # If we can't identify it
    raise ValueError("Unrecognized format")
