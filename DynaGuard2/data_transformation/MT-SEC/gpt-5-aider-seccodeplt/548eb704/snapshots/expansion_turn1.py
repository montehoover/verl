from typing import Optional

def validate_and_cleanup(text_input: str) -> Optional[str]:
    """
    Strip leading and trailing whitespace from the input string.
    Return the cleaned string if non-empty; otherwise return None.
    """
    if not isinstance(text_input, str):
        return None
    cleaned = text_input.strip()
    return cleaned if cleaned != "" else None
