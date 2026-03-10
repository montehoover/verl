import re

def validate_operation_input(input_string):
    """
    Validates if a string contains only numbers, basic arithmetic operators (+, -, *, /), and spaces.
    
    Args:
        input_string (str): The string to validate
        
    Returns:
        bool: True if the input is valid, False otherwise
    """
    # Pattern allows digits, operators (+, -, *, /), spaces, and decimal points
    pattern = r'^[0-9+\-*/.\s]+$'
    
    if not input_string or not input_string.strip():
        return False
    
    return bool(re.match(pattern, input_string))
