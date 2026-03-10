import re

def count_words(text_content):
    if not text_content:
        return 0
    return len(re.findall(r"\b\w+\b", text_content))
