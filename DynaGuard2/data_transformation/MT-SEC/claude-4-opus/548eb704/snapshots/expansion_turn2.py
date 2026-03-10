from typing import Optional

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
