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


def parse_expression(expression):
    """
    Parses a validated mathematical expression into tokens.
    
    Args:
        expression (str): The validated mathematical expression
        
    Returns:
        list: A list of tokens (numbers, operators, parentheses)
    """
    # Remove all whitespace
    expression = expression.replace(' ', '')
    
    # Pattern to match numbers (including decimals) and operators/parentheses
    token_pattern = r'(\d+\.?\d*|[+\-*/()])'
    
    # Find all tokens
    tokens = re.findall(token_pattern, expression)
    
    # Convert numeric strings to float/int
    parsed_tokens = []
    for token in tokens:
        if token in '+-*/()':
            parsed_tokens.append(token)
        else:
            # Convert to appropriate numeric type
            if '.' in token:
                parsed_tokens.append(float(token))
            else:
                parsed_tokens.append(int(token))
    
    return parsed_tokens
