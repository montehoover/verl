import re

def sanitize_input(input_string):
    """
    Check if a string only contains digits, basic arithmetic operators (+, -, *, /), and spaces.
    
    Args:
        input_string (str): The string to validate
        
    Returns:
        bool: True if the input only contains allowed characters, False otherwise
    """
    # Define the pattern for allowed characters
    # This pattern matches strings that only contain:
    # - digits (0-9)
    # - arithmetic operators (+, -, *, /)
    # - spaces
    allowed_pattern = r'^[0-9+\-*/\s]+$'
    
    # Check if the input string matches the pattern
    if re.match(allowed_pattern, input_string):
        return True
    else:
        return False
