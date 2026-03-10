import json
from typing import Union

def unpack_serialized_object(byte_data: Union[bytes, bytearray, memoryview], encoding: str = "utf-8") -> str:
    """
    Decode binary data using the given text encoding (default UTF-8).
    If the decoded text is valid JSON, return a pretty-printed JSON string.
    Otherwise, return the decoded text as-is.

    Parameters:
        byte_data: bytes-like object containing the data.
        encoding: text encoding to use for decoding (defaults to 'utf-8').

    Returns:
        A readable string suitable for logging.
    """
    if not isinstance(byte_data, (bytes, bytearray, memoryview)):
        raise TypeError("byte_data must be a bytes-like object")

    data = bytes(byte_data)

    try:
        text = data.decode(encoding)
    except UnicodeDecodeError:
        # Ensure we always return a readable string
        text = data.decode(encoding, errors="replace")

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return text

    return json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=True)
