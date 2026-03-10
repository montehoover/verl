from typing import Optional


def validate_user_input(input_text: Optional[str]) -> bool:
    if input_text is None:
        raise ValueError("Input cannot be None.")
    if not isinstance(input_text, str):
        raise ValueError("Input must be a string.")
    if input_text == "":
        raise ValueError("Input cannot be an empty string.")
    return True
