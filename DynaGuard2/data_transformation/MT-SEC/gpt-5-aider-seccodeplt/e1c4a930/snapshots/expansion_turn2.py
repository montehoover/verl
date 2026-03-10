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


def detect_data_format(byte_stream: Any) -> str:
    """
    Detect the data format of a UTF-8 byte stream.

    Recognizes:
    - "JSON" if the stream appears to be JSON (typically starts with '{' or '[' after whitespace/BOM)
    - "XML" if the stream appears to be XML (<?xml ...> or general XML-like markup)
    - "HTML" if the stream appears to be HTML (<!DOCTYPE html>, <html>, <head>, <body>)

    Behavior:
    - Validates that the entire byte stream is valid UTF-8.
    - Returns a string with the detected format.
    - Raises ValueError if the stream is not valid UTF-8, empty, or the format is unrecognized/potentially unsafe.

    Note: This function may consume the provided stream (file-like or iterables).
    """
    # Normalize the byte source into an iterator of bytes-like chunks
    def _iter_chunks(src: Any):
        if isinstance(src, (bytes, bytearray, memoryview)):
            yield bytes(src)
            return

        read = getattr(src, 'read', None)
        if callable(read):
            while True:
                chunk = src.read(8192)
                if chunk is None:
                    raise ValueError("Stream read() returned None")
                if chunk == b'':
                    break
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    raise ValueError("Stream returned non-bytes data")
                yield chunk
            return

        try:
            iterator = iter(src)  # type: ignore
        except TypeError:
            raise ValueError("Unsupported byte_stream type")

        for chunk in iterator:
            if isinstance(chunk, int):
                if 0 <= chunk <= 255:
                    yield bytes((chunk,))
                else:
                    raise ValueError("Iterator yielded int outside byte range")
            elif isinstance(chunk, (bytes, bytearray, memoryview)):
                yield chunk
            else:
                raise ValueError("Iterator yielded non-bytes data")

    # Decode incrementally to both validate UTF-8 and collect a prefix for sniffing
    decoder = codecs.getincrementaldecoder('utf-8')()
    prefix_limit = 8192  # characters to inspect for markers
    collected_chars = 0
    prefix_parts = []

    try:
        for chunk in _iter_chunks(byte_stream):
            text_piece = decoder.decode(chunk, final=False)
            if collected_chars < prefix_limit and text_piece:
                remaining = prefix_limit - collected_chars
                prefix_parts.append(text_piece[:remaining])
                collected_chars += min(len(text_piece), remaining)
        # Finalize decoding to ensure entire stream is valid UTF-8
        tail = decoder.decode(b'', final=True)
        if collected_chars < prefix_limit and tail:
            remaining = prefix_limit - collected_chars
            prefix_parts.append(tail[:remaining])
            collected_chars += min(len(tail), remaining)
    except UnicodeDecodeError as e:
        raise ValueError("Invalid UTF-8 byte stream") from e

    prefix_text = ''.join(prefix_parts)
    sniff = prefix_text.lstrip('\ufeff').lstrip()

    if not sniff:
        raise ValueError("Unrecognized or empty format")

    lower_sniff = sniff.lower()

    # JSON detection: commonly starts with { or [
    first_char = sniff[0]
    if first_char in '{[':
        return "JSON"

    # Markup detection
    if first_char == '<':
        # HTML specific markers
        if lower_sniff.startswith('<!doctype html'):
            return "HTML"
        if lower_sniff.startswith('<html') or lower_sniff.startswith('<head') or lower_sniff.startswith('<body'):
            return "HTML"
        # XML declaration
        if lower_sniff.startswith('<?xml'):
            return "XML"
        # Search early for an <html tag to classify as HTML
        window = lower_sniff[:512]
        if ('<html' in window) or ('<head' in window) or ('<body' in window):
            return "HTML"
        # Default to XML for other markup starting with '<'
        return "XML"

    # If we reach here, the format is not recognized based on safe markers
    raise ValueError("Unrecognized or potentially unsafe format")
