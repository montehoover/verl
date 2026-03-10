from typing import Optional

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
