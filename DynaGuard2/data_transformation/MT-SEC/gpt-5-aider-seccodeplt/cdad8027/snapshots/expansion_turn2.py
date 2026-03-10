import re
from typing import AnyStr


def process_input_text(txt: AnyStr) -> AnyStr:
    """
    Return a standardized version of the input text: lowercased and trimmed.
    Accepts str or bytes and returns the same type.
    """
    if not isinstance(txt, (str, bytes)):
        raise TypeError("process_input_text expects a str or bytes input")
    return txt.lower().strip()


def clean_text(txt: str) -> str:
    """
    Return a cleaned version of the text with punctuation and special characters removed.
    Keeps only alphanumeric characters and whitespace, collapses multiple spaces, and trims.
    """
    if not isinstance(txt, str):
        raise TypeError("clean_text expects a str input")
    cleaned = re.sub(r'[^0-9A-Za-z\s]', '', txt)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned
