import codecs
from typing import Any


def validate_byte_stream(byte_stream: Any) -> bool:
    """
    Returns True if the given byte_stream contains only well-formed UTF-8 sequences.
    Accepts:
      - bytes, bytearray, memoryview, or any object supporting the buffer protocol
      - file-like objects with a .read() method that returns bytes
    """
    try:
        # If it's a file-like object, validate incrementally without loading everything into memory.
        if hasattr(byte_stream, "read"):
            decoder = codecs.getincrementaldecoder("utf-8")()
            while True:
                chunk = byte_stream.read(8192)
                if not chunk:
                    break
                if not isinstance(chunk, (bytes, bytearray)):
                    # Not bytes-like; invalid for a byte stream
                    return False
                decoder.decode(chunk)
            # Flush the decoder to ensure no incomplete sequence remains.
            decoder.decode(b"", final=True)
            return True

        # Otherwise, treat it as a bytes-like object.
        if isinstance(byte_stream, memoryview):
            data = byte_stream.tobytes()
        elif isinstance(byte_stream, (bytes, bytearray)):
            data = byte_stream
        else:
            # Attempt to coerce any bytes-like object to bytes.
            data = bytes(byte_stream)

        # Strict decoding will raise UnicodeDecodeError for any malformed UTF-8.
        if isinstance(data, bytearray):
            data = bytes(data)
        data.decode("utf-8")
        return True
    except (UnicodeDecodeError, TypeError, ValueError):
        return False
