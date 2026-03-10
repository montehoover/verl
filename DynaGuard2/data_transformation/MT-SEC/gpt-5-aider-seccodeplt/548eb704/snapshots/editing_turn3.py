import re

def parse_text_input(provided_input: str) -> list[str]:
    if not isinstance(provided_input, str):
        raise ValueError("provided_input must be a string")
    try:
        words = re.findall(r"\b\w+\b", provided_input, flags=re.UNICODE)
        return words
    except re.error as e:
        raise ValueError(f"Failed to parse text input: {e}") from e
