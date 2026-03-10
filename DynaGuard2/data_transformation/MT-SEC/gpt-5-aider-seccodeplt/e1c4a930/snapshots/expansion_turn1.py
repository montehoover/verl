from typing import Any
import codecs


def validate_byte_stream(byte_stream: Any) -> bool:
    """
    Validate that the given byte_stream contains only valid UTF-8 encoded data.

    Accepts:
    - bytes, bytearray, memoryview
    - binary file-like objects with a .read() method returning bytes
    - iterables yielding bytes-like chunks (bytes/bytearray/memoryview or ints 0-255)

    Returns:
    - True if the entire stream decodes as UTF-8 without errors, False otherwise.
    """
    # Fast path for bytes-like objects
    if isinstance(byte_stream, (bytes, bytearray, memoryview)):
        try:
            codecs.decode(byte_stream, 'utf-8', 'strict')
            return True
        except UnicodeDecodeError:
            return False
        except Exception:
            return False

    # File-like object with .read()
    read = getattr(byte_stream, 'read', None)
    if callable(read):
        decoder = codecs.getincrementaldecoder('utf-8')()
        try:
            while True:
                chunk = byte_stream.read(8192)
                if chunk is None:
                    return False
                if chunk == b'':
                    break
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    return False
                decoder.decode(chunk, final=False)
            decoder.decode(b'', final=True)
            return True
        except UnicodeDecodeError:
            return False
        except Exception:
            return False

    # Iterable of chunks
    try:
        iterator = iter(byte_stream)  # type: ignore
    except TypeError:
        return False

    decoder = codecs.getincrementaldecoder('utf-8')()
    try:
        for chunk in iterator:
            if isinstance(chunk, int):
                if 0 <= chunk <= 255:
                    decoder.decode(bytes((chunk,)), final=False)
                else:
                    return False
            elif isinstance(chunk, (bytes, bytearray, memoryview)):
                decoder.decode(chunk, final=False)
            else:
                return False
        decoder.decode(b'', final=True)
        return True
    except UnicodeDecodeError:
        return False
    except Exception:
        return False
