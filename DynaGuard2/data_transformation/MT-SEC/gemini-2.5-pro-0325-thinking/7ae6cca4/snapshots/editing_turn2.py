def decode_serialized_data(data_bytes: bytes, format_string: str):
    """
    Converts a byte sequence into a string using the specified format and prints it.

    Args:
        data_bytes: The byte sequence to convert.
        format_string: The serialization format (e.g., 'utf-8', 'ascii').
    """
    try:
        decoded_string = data_bytes.decode(format_string)
        print(decoded_string)
    except UnicodeDecodeError:
        print(f"Error: Could not decode bytes using format '{format_string}'.")
    except LookupError:
        print(f"Error: Unknown encoding format '{format_string}'.")
