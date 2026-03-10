import re

def clean_text(text: str) -> str:
    return re.sub(r'\W+', ' ', text).strip()

def get_user_input() -> str:
    return input("Please enter your text: ")

def parse_user_input(text: str) -> list:
    try:
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")
        words = re.findall(r'\b\w+\b', text)
        return words
    except Exception as e:
        raise ValueError(f"Error processing input: {e}")
