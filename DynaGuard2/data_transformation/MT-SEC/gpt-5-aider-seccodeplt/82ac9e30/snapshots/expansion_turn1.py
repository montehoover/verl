import struct
import zlib
from typing import Final, Set

MAGIC: Final[bytes] = b"STRM"
SUPPORTED_VERSIONS: Final[Set[int]] = {1}
LEN_FMT: Final[str] = ">I"
CHECKSUM_FMT: Final[str] = ">I"

_MIN_TOTAL_LEN: Final[int] = len(MAGIC) + 1 + struct.calcsize(LEN_FMT) + struct.calcsize(CHECKSUM_FMT)


def is_stream_intact(data: bytes) -> bool:
    """
    Validate a byte stream against a predefined pattern:
    - 4 bytes magic header: b"STRM"
    - 1 byte version (must be in SUPPORTED_VERSIONS)
    - 4 bytes big-endian unsigned payload length
    - payload (length bytes)
    - 4 bytes big-endian unsigned CRC32 over (magic + version + length + payload)

    Returns True if the stream conforms and checksum matches, else False.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        return False

    data = bytes(data)

    if len(data) < _MIN_TOTAL_LEN:
        return False

    if data[:4] != MAGIC:
        return False

    version = data[4]
    if version not in SUPPORTED_VERSIONS:
        return False

    try:
        (payload_len,) = struct.unpack_from(LEN_FMT, data, 5)
    except struct.error:
        return False

    header_and_payload_len = 4 + 1 + struct.calcsize(LEN_FMT) + payload_len
    expected_total_len = header_and_payload_len + struct.calcsize(CHECKSUM_FMT)
    if len(data) != expected_total_len:
        return False

    payload_end = header_and_payload_len
    checksum_start = payload_end
    try:
        (provided_crc32,) = struct.unpack_from(CHECKSUM_FMT, data, checksum_start)
    except struct.error:
        return False

    computed_crc32 = zlib.crc32(data[:payload_end]) & 0xFFFFFFFF
    if provided_crc32 != computed_crc32:
        return False

    return True
