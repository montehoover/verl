import struct
import io

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
