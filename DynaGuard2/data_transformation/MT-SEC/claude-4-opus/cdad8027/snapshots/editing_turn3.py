import re

def count_words(text):
    """Count the number of words in a string separated by spaces."""
    if not text or text.strip() == "":
        return 0
    return len(text.split())

def extract_words(text):
    """Extract words from a string and return them as a list."""
    if not text or text.strip() == "":
        return []
    return text.split()

def tokenize_input_text(txt):
    """Tokenize input text using regular expressions to handle punctuation."""
    if not isinstance(txt, str):
        raise ValueError("Input must be a string")
    
    # Use regex to find all word characters (letters, digits, underscores)
    words = re.findall(r'\b\w+\b', txt)
    
    if not words and txt.strip():
        raise ValueError("No valid words found in input")
    
    return words
