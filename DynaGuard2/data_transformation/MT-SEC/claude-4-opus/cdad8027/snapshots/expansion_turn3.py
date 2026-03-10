from typing import AnyStr
import re

def process_input_text(txt: AnyStr) -> str:
    """
    Process input text by converting it to lowercase for standardization.
    
    Args:
        txt: Input text to process
        
    Returns:
        Lowercased version of the input text
    """
    if isinstance(txt, bytes):
        txt = txt.decode('utf-8')
    return txt.lower()

def clean_text(txt: str) -> str:
    """
    Clean text by removing punctuation and special characters.
    
    Args:
        txt: Input text to clean
        
    Returns:
        Cleaned text with only alphanumeric characters and spaces
    """
    # Remove all non-alphanumeric characters except spaces
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', txt)
    # Replace multiple spaces with single space
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Strip leading and trailing whitespace
    return cleaned.strip()

def tokenize_input_text(txt: str) -> list[str]:
    """
    Tokenize input text into individual words using regular expressions.
    
    Args:
        txt: Input text to tokenize
        
    Returns:
        List of words extracted from the text
        
    Raises:
        ValueError: If complications arise during processing
    """
    try:
        # Extract words using regex pattern for word boundaries
        words = re.findall(r'\b\w+\b', txt)
        return words
    except Exception as e:
        raise ValueError(f"Error during tokenization: {str(e)}")
