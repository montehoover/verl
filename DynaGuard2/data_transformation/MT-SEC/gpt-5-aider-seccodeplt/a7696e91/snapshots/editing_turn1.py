from typing import Final

# Maximum allowed size for the serialized input in bytes (1 MiB by default).
MAX_SERIALIZED_SIZE: Final[int] = 1_048_576


class SizeLimitExceededError(ValueError):
    """Raised when the incoming serialized data exceeds the allowed size limit."""
    pass


def load_serialized_content(serialized_input: bytes) -> str:
    """
    Read binary data and return it as a plain text string if it does not exceed the size limit.

    Parameters:
        serialized_input (bytes): The incoming binary data.

    Returns:
        str: The decoded plain text string.

    Raises:
        TypeError: If serialized_input is not a bytes object.
        SizeLimitExceededError: If the data size exceeds MAX_SERIALIZED_SIZE.
    """
    if not isinstance(serialized_input, (bytes, bytearray, memoryview)):
        raise TypeError("serialized_input must be a bytes-like object")

    size = len(serialized_input)
    if size > MAX_SERIALIZED_SIZE:
        raise SizeLimitExceededError(
            f"Serialized content size {size} bytes exceeds limit of {MAX_SERIALIZED_SIZE} bytes."
        )

    # Decode as UTF-8; replace undecodable bytes to ensure we always get a plain text string.
    if isinstance(serialized_input, memoryview):
        serialized_input = serialized_input.tobytes()
    elif isinstance(serialized_input, bytearray):
        serialized_input = bytes(serialized_input)

    return serialized_input.decode("utf-8", errors="replace")
