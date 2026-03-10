from typing import Any, Optional
import hashlib

__all__ = ["deserialize_data"]

def deserialize_data(
    data: Any,
    encoding: str = "utf-8",
    errors: str = "strict",
    expected_hash: Optional[str] = None,
) -> str:
    """
    Convert a byte-like data stream into a string using simple decoding.
    Optionally verify integrity by comparing the SHA-256 hash of the raw bytes
    (or the encoded string if 'data' is a str) against 'expected_hash'.

    Parameters:
        data: A bytes-like object (e.g., bytes, bytearray, memoryview) or str.
        encoding: Text encoding to use (default: 'utf-8').
        errors: Error handling scheme (default: 'strict').
        expected_hash: Hex-encoded SHA-256 hash to verify against. If provided,
            a mismatch returns an error string.

    Returns:
        The decoded string, or an error string if hash verification fails.

    Raises:
        TypeError: If 'data' is not bytes-like or str.
        UnicodeDecodeError: If decoding fails and errors='strict'.
    """
    if isinstance(data, str):
        b = data.encode(encoding, errors=errors)
        decoded = data
    else:
        try:
            b = bytes(data)
        except TypeError:
            raise TypeError("deserialize_data expects a bytes-like object or str") from None
        decoded = b.decode(encoding, errors=errors)

    if expected_hash is not None:
        actual_hash = hashlib.sha256(b).hexdigest()
        if actual_hash.lower() != expected_hash.strip().lower():
            return "ERROR: hash verification failed"

    return decoded
