from typing import Iterable, Union, BinaryIO, Iterator, Any
import codecs
import io


def validate_byte_stream(stream: Union[bytes, bytearray, memoryview, Iterable[bytes], BinaryIO]) -> bool:
    """
    Validate whether the provided byte stream contains valid UTF-8 encoded data.

    The 'stream' parameter may be:
    - A bytes-like object (bytes, bytearray, memoryview)
    - An iterable yielding bytes-like chunks
    - A binary file-like object (e.g., an object with a .read() method)

    Returns:
        bool: True if the entire stream is valid UTF-8, False otherwise.
    """
    def iter_chunks(src: Any) -> Iterator[bytes]:
        # Single bytes-like object
        if isinstance(src, (bytes, bytearray, memoryview)):
            yield bytes(src)
            return

        # File-like binary object with .read()
        if hasattr(src, "read"):
            # Prefer binary IO base types
            if isinstance(src, io.TextIOBase):
                raise TypeError("Expected a binary stream, received a text stream.")
            while True:
                chunk = src.read(8192)
                if chunk is None:
                    # Treat None like EOF for non-blocking streams
                    break
                if not chunk:
                    break
                if isinstance(chunk, str):
                    raise TypeError("Expected bytes from binary stream, got str.")
                yield bytes(chunk)
            return

        # Iterable of chunks
        if isinstance(src, Iterable):
            for chunk in src:
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    raise TypeError("Iterable must yield bytes-like objects.")
                yield bytes(chunk)
            return

        raise TypeError("Unsupported stream type for UTF-8 validation.")

    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    try:
        for chunk in iter_chunks(stream):
            decoder.decode(chunk, final=False)
        # Finalize to detect any incomplete sequences at the end
        decoder.decode(b"", final=True)
        return True
    except UnicodeDecodeError:
        return False
