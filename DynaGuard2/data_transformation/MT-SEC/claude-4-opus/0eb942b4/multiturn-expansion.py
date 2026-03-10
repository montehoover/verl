from typing import List
import re

def split_string(text: str) -> List[str]:
    """Split a string into a list of words."""
    return text.split()

def clean_and_split_string(text: str) -> List[str]:
    """Clean and split a string into a list of words.
    
    Removes leading/trailing whitespace and converts to lowercase.
    """
    return text.strip().lower().split()

def analyze_user_string(text: str) -> List[str]:
    """Analyze user string and return list of words.
    
    Raises ValueError if input processing encounters issues.
    """
    try:
        # Use regex to find all word characters
        words = re.findall(r'\w+', text)
        if not words and text.strip():
            raise ValueError("No valid words found in input")
        return words
    except Exception as e:
        raise ValueError(f"Error processing input: {str(e)}")
