from typing import AnyStr

def process_input_text(txt: AnyStr) -> AnyStr:
    """
    Converts the input text to a standardized format (lowercase).

    Args:
        txt: The input string.

    Returns:
        The lowercased string.
    """
    if isinstance(txt, str):
        return txt.lower()
    elif isinstance(txt, bytes):
        return txt.lower()
    else:
        # Or raise a TypeError, depending on desired handling for other types
        return txt
