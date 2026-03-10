# Define a constant for the size limit (e.g., 1MB)
MAX_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB

class SizeLimitExceededError(ValueError):
    """Custom exception for when input data exceeds the size limit."""
    pass

def load_serialized_content(serialized_input: bytes) -> str:
    """
    Reads binary data, checks its size against a limit, and converts it to a string.

    Args:
        serialized_input: A bytes object containing the serialized data.

    Returns:
        A plain text string if the data is within the size limit.

    Raises:
        SizeLimitExceededError: If the binary data is too large.
        UnicodeDecodeError: If the binary data cannot be decoded to UTF-8.
    """
    if not isinstance(serialized_input, bytes):
        raise TypeError("Input must be a bytes object.")

    if len(serialized_input) > MAX_SIZE_BYTES:
        raise SizeLimitExceededError(
            f"Input data size ({len(serialized_input)} bytes) "
            f"exceeds the limit of {MAX_SIZE_BYTES} bytes."
        )

    try:
        # Assuming UTF-8 encoding for plain text conversion
        return serialized_input.decode('utf-8')
    except UnicodeDecodeError as e:
        # Re-raise with a more specific message or handle as needed
        raise UnicodeDecodeError(e.encoding, e.object, e.start, e.end, "Failed to decode input as UTF-8.")
