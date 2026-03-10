import zlib
from typing import Union


def validate_byte_stream(stream: Union[bytes, bytearray, memoryview]) -> bool:
    """
    Validate a byte stream.

    Rules:
    - Must be a non-empty bytes-like object.
    - Considered invalid if the stream is entirely 0x00 or entirely 0xFF (common corrupted/sentinel patterns).
    - If the stream appears to end with a 4-byte CRC32 trailer of the preceding payload, and it matches (either endianness),
      it is considered valid.
    - Otherwise, if non-empty and not a degenerate pattern, it is considered valid (no definitive corruption detected).
    """
    try:
        buf = memoryview(stream).tobytes()
    except TypeError:
        return False

    n = len(buf)
    if n == 0:
        return False

    # Obvious corruption/placeholder patterns
    if all(b == 0x00 for b in buf):
        return False
    if all(b == 0xFF for b in buf):
        return False

    # Optional CRC32 trailer verification (payload + 4-byte CRC)
    if n >= 5:
        payload = buf[:-4]
        trailer = buf[-4:]
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        if trailer == crc.to_bytes(4, 'big') or trailer == crc.to_bytes(4, 'little'):
            return True

    # No definitive corruption detected
    return True
