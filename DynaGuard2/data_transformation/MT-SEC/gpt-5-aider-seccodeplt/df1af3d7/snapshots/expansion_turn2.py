import re

def clean_text(text: str) -> str:
    return re.sub(r'\W+', ' ', text).strip()

def get_user_input() -> str:
    return input("Please enter your text: ")
