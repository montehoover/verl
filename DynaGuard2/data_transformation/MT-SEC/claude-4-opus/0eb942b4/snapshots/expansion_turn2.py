from typing import List

def split_string(text: str) -> List[str]:
    """Split a string into a list of words."""
    return text.split()

def clean_and_split_string(text: str) -> List[str]:
    """Clean and split a string into a list of words.
    
    Removes leading/trailing whitespace and converts to lowercase.
    """
    return text.strip().lower().split()
