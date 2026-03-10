import re

def count_words(text):
    if not text or text.strip() == "":
        return 0
    return len(text.split())

def extract_words(text):
    if not text or text.strip() == "":
        return []
    return text.split()

def analyze_user_string(input_text):
    if not isinstance(input_text, str):
        raise ValueError("Input must be a string")
    
    words = re.findall(r'\b\w+\b', input_text)
    
    if not words:
        raise ValueError("No valid words found in input")
    
    return words
