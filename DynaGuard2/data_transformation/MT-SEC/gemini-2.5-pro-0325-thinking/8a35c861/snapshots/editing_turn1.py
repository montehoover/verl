def simple_shift(text: str) -> str:
    """
    Applies a Caesar cipher shift to the input string using a fixed key.

    Args:
        text: The input string to be shifted.

    Returns:
        The Caesar shifted string.
    """
    key = 3  # Fixed key for the Caesar cipher
    shifted_text = []

    for char_code in [ord(c) for c in text]:
        # Uppercase letters
        if 65 <= char_code <= 90:  # ord('A') to ord('Z')
            shifted_char_code = (char_code - 65 + key) % 26 + 65
        # Lowercase letters
        elif 97 <= char_code <= 122:  # ord('a') to ord('z')
            shifted_char_code = (char_code - 97 + key) % 26 + 97
        # Other characters
        else:
            shifted_char_code = char_code
        shifted_text.append(chr(shifted_char_code))

    return "".join(shifted_text)
