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


def tokenize_input_text(txt: str) -> list[str]:
    """
    Tokenize the input text into words using regular expressions.
    Returns a list of alphanumeric word tokens (ASCII letters and digits).
    Raises:
        TypeError: if txt is not a string.
        ValueError: if an error occurs during regex processing.
    """
    if not isinstance(txt, str):
        raise TypeError("tokenize_input_text expects a str input")
    try:
        tokens = re.findall(r'[0-9A-Za-z]+', txt)
    except Exception as e:
        raise ValueError(f"Tokenization failed due to processing error: {e}") from e
    return tokens
