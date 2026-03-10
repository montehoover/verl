import re
from typing import Optional


def validate_user_input(input_text: Optional[str]) -> bool:
    if input_text is None:
        raise ValueError("Input cannot be None.")
    if not isinstance(input_text, str):
        raise ValueError("Input must be a string.")
    if input_text == "":
        raise ValueError("Input cannot be an empty string.")
    return True


def count_words(input_text: str) -> int:
    """
    Count the number of words in the given input_text.
    A word is defined as a sequence of non-whitespace characters separated by whitespace.
    Raises ValueError if the input is invalid according to validate_user_input.
    """
    validate_user_input(input_text)
    return len(input_text.split())


def transform_user_text(text_input: str) -> list[str]:
    """
    Parse the input text and extract words using regular expressions.
    Returns a list of words. Raises ValueError if processing fails or input is invalid.
    """
    validate_user_input(text_input)
    try:
        return re.findall(r"\b\w+\b", text_input, flags=re.UNICODE)
    except re.error as e:
        raise ValueError(f"Regex processing error: {e}") from e
    except Exception as e:
        raise ValueError(f"Error processing input: {e}") from e
