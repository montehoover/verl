import re

def sanitize_input(expression):
    """
    Validates that a string contains only valid mathematical expression characters.
    
    Args:
        expression (str): The input string to validate
        
    Returns:
        bool: True if the input contains only valid characters
        
    Raises:
        ValueError: If invalid characters are found
    """
    # Define the pattern for valid characters: digits, operators, parentheses, decimal points, and spaces
    valid_pattern = r'^[0-9+\-*/().\s]+$'
    
    if not expression:
        return True
    
    if re.match(valid_pattern, expression):
        return True
    else:
        # Find invalid characters for the error message
        invalid_chars = set(re.findall(r'[^0-9+\-*/().\s]', expression))
        raise ValueError(f"Invalid characters found: {', '.join(invalid_chars)}")
