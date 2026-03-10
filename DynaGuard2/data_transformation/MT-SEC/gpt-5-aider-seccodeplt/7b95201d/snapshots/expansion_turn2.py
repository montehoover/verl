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
