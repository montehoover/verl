from typing import Optional
import re

def validate_user_input(input_text: Optional[str]) -> bool:
    """
    Validates if the input text is a valid string to process.
    
    Args:
        input_text: The string to validate
        
    Returns:
        bool: True if the input is valid
        
    Raises:
        ValueError: If the input is None or empty
    """
    if input_text is None:
        raise ValueError("Input cannot be None")
    
    if not input_text.strip():
        raise ValueError("Input cannot be empty or contain only whitespace")
    
    return True


def count_words(input_text: str) -> int:
    """
    Counts the number of words in the input text.
    
    Args:
        input_text: The string to count words in
        
    Returns:
        int: The number of words in the string
    """
    return len(input_text.split())


def transform_user_text(text_input: str) -> list[str]:
    """
    Parses a user's string input and extracts each word into a list.
    
    Args:
        text_input: The string to parse and extract words from
        
    Returns:
        list[str]: A list containing the extracted words
        
    Raises:
        ValueError: If there is a problem during input processing
    """
    try:
        # Use regex to find all word characters (letters, digits, underscores)
        words = re.findall(r'\b\w+\b', text_input)
        return words
    except Exception as e:
        raise ValueError(f"Error processing input: {str(e)}")
