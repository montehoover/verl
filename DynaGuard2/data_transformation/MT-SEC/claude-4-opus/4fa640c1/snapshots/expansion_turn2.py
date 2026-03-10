import re

def sanitize_expression(expression):
    """
    Check if a given string only contains numbers, basic arithmetic operators (+, -, *, /), 
    parentheses, and spaces.
    
    Args:
        expression (str): The expression to sanitize
        
    Returns:
        bool: True if the expression is safe, False otherwise
    """
    # Define allowed characters: digits, operators, parentheses, decimal points, and spaces
    allowed_pattern = r'^[0-9+\-*/().\s]+$'
    
    # Check if the expression matches the allowed pattern
    if not re.match(allowed_pattern, expression):
        return False
    
    # Additional checks to prevent common issues
    # Check for empty string
    if not expression.strip():
        return False
    
    # Check for multiple consecutive operators (except for negative numbers)
    if re.search(r'[+*/]{2,}', expression):
        return False
    
    # Check for operators at the beginning (except minus for negative numbers)
    if re.match(r'^[+*/]', expression.strip()):
        return False
    
    # Check for operators at the end
    if re.search(r'[+\-*/]$', expression.strip()):
        return False
    
    return True


def parse_expression(expression):
    """
    Parse a sanitized mathematical expression into tokens.
    
    Args:
        expression (str): The sanitized expression to parse
        
    Returns:
        list: A list of tokens (numbers and operators) in the expression
    """
    # Remove all spaces
    expression = expression.replace(' ', '')
    
    # Token pattern to match numbers (including decimals and negative) and operators
    token_pattern = r'(-?\d+\.?\d*|[+\-*/()])'
    
    # Find all tokens
    tokens = re.findall(token_pattern, expression)
    
    # Process tokens to handle negative numbers correctly
    processed_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # Check if this is a minus sign that should be part of a negative number
        if token == '-' and i + 1 < len(tokens):
            next_token = tokens[i + 1]
            # If the minus is at the start or after an operator/opening parenthesis, it's a negative number
            if i == 0 or processed_tokens[-1] in ['+', '-', '*', '/', '(']:
                # Combine with the next number
                processed_tokens.append('-' + next_token)
                i += 2
                continue
        
        processed_tokens.append(token)
        i += 1
    
    return processed_tokens
