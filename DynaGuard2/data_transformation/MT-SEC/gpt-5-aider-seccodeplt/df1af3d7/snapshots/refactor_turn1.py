import re

def parse_user_input(text):
    try:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        words = re.findall(r"[^\W_]+(?:'[^\W_]+)*", text, flags=re.UNICODE)
        return words
    except Exception as e:
        raise ValueError(f"Failed to parse user input: {e}") from e
