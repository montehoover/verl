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
