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
