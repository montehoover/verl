from typing import IO, Iterable, Union, Iterator

ByteLike = Union[bytes, bytearray, memoryview]
ByteItem = Union[int, ByteLike]
ByteStream = Union[ByteLike, IO[bytes], Iterable[ByteItem]]

_ALLOWED_EXTRA_WHITESPACE = {9, 10, 13}  # tab, LF, CR


def _iter_byte_chunks(stream: ByteStream, chunk_size: int = 8192) -> Iterator[bytes]:
    """
    Yield bytes chunks from various kinds of byte streams:
    - bytes/bytearray/memoryview: yielded as a single chunk
    - file-like object with read(): read in chunks
    - iterable of ints (0-255) or bytes-like: yielded piecewise
    """
    if isinstance(stream, (bytes, bytearray, memoryview)):
        yield bytes(stream)
        return

    read = getattr(stream, "read", None)
    if callable(read):
        while True:
            chunk = read(chunk_size)
            if not chunk:
                break
            if not isinstance(chunk, (bytes, bytearray, memoryview)):
                raise TypeError("read() must return bytes-like data for a byte stream")
            yield bytes(chunk)
        return

    if isinstance(stream, Iterable):
        for part in stream:
            if isinstance(part, int):
                if 0 <= part <= 255:
                    yield bytes((part,))
                else:
                    raise ValueError("Iterable provided byte value out of range (0-255)")
            elif isinstance(part, (bytes, bytearray, memoryview)):
                yield bytes(part)
            else:
                raise TypeError("Iterable must yield ints (0-255) or bytes-like objects")
        return

    raise TypeError("Unsupported byte stream type")


def is_printable_byte_stream(stream: ByteStream) -> bool:
    """
    Return True if the given byte stream consists entirely of printable ASCII characters.
    Allowed bytes:
      - ASCII printable range: 32..126
      - Common whitespace: tab (9), line feed (10), carriage return (13)
    """
    for chunk in _iter_byte_chunks(stream):
        for b in chunk:
            if not (32 <= b <= 126 or b in _ALLOWED_EXTRA_WHITESPACE):
                return False
    return True
