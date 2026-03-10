import json

def unpack_serialized_object(byte_data: bytes) -> str:
    """
    Unpacks binary data, interpreting it as UTF-8 encoded text.
    Checks if the text is a JSON string and returns the decoded text.

    Args:
        byte_data: A bytes object, assumed to contain UTF-8 encoded text.

    Returns:
        The decoded string. If byte_data is not valid UTF-8,
        a UnicodeDecodeError will be raised during decoding.
    """
    decoded_text = byte_data.decode('utf-8')
    
    try:
        # Attempt to parse as JSON to check its structure.
        # The content of the JSON is not used beyond this check for now.
        json.loads(decoded_text)
        # If parsing succeeds, it's a JSON string.
        # The function returns the original decoded_text (the JSON string).
    except json.JSONDecodeError:
        # If parsing fails, it's not a JSON string (it's regular text).
        # The function returns the decoded_text.
        pass
        
    return decoded_text
