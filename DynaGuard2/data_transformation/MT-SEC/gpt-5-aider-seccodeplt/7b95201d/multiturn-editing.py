import re

def transform_user_text(text_input):
    if not isinstance(text_input, str):
        raise ValueError("text_input must be a string")
    try:
        words = re.findall(r"\b\w+\b", text_input)
    except Exception as exc:
        raise ValueError("Error processing input") from exc
    return words
