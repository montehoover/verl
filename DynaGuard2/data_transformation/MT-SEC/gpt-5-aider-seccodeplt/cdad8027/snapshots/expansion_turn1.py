from typing import AnyStr


def process_input_text(txt: AnyStr) -> AnyStr:
    """
    Return a standardized version of the input text: lowercased and trimmed.
    Accepts str or bytes and returns the same type.
    """
    if not isinstance(txt, (str, bytes)):
        raise TypeError("process_input_text expects a str or bytes input")
    return txt.lower().strip()
