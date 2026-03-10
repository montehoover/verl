from typing import Optional
import re

def validate_and_cleanup(text_input: str) -> Optional[str]:
    """
    Clean up and validate raw input strings.
    
    Args:
        text_input: The raw input string to process
        
    Returns:
        Cleaned string with stripped whitespace, or None if empty
    """
    if not text_input:
        return None
    
    cleaned = text_input.strip()
    
    if not cleaned:
        return None
    
    return cleaned


def normalize_text_case(cleaned_text: str) -> str:
    """
    Convert all characters in the text to lowercase.
    
    Args:
        cleaned_text: The cleaned text string to normalize
        
    Returns:
        The text with all characters converted to lowercase
    """
    return cleaned_text.lower()


def parse_text_input(provided_input: str) -> list[str]:
    """
    Extract individual words from the text input using regular expressions.
    
    Args:
        provided_input: The input string to parse
        
    Returns:
        A list of words extracted from the input
        
    Raises:
        ValueError: If problems occur during processing
    """
    try:
        if not provided_input:
            raise ValueError("Input cannot be empty")
        
        # Extract words using regex pattern that matches word characters
        words = re.findall(r'\b\w+\b', provided_input)
        
        if not words:
            raise ValueError("No words found in the input")
        
        return words
    except Exception as e:
        raise ValueError(f"Error processing text: {str(e)}")
