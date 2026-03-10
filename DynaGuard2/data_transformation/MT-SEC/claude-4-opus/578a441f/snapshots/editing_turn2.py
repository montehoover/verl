import re

def analyze_script(script: str) -> list:
    # Define security-sensitive characters
    security_sensitive = ['<', '>', '&', '|', ';', '$', '`', '\\', '"', "'", '\n', '\r', '\0']
    
    # Check for security-sensitive characters
    for char in security_sensitive:
        if char in script:
            raise ValueError(f"Security-sensitive character '{char}' found in script")
    
    # Define token pattern - matches words, numbers, and common operators
    token_pattern = r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|\+|\-|\*|\/|\=|\(|\)|\{|\}|\[|\]|\,|\.'
    
    # Find all tokens
    tokens = re.findall(token_pattern, script)
    
    # Define approved syntax elements
    approved_keywords = {'if', 'else', 'elif', 'for', 'while', 'def', 'class', 'return', 'import', 'from', 'as', 'in', 'is', 'and', 'or', 'not', 'True', 'False', 'None', 'pass', 'break', 'continue', 'try', 'except', 'finally', 'with', 'lambda', 'yield'}
    approved_operators = {'+', '-', '*', '/', '=', '(', ')', '{', '}', '[', ']', ',', '.'}
    
    # Validate and filter tokens
    validated_tokens = []
    for token in tokens:
        # Check if token is a number
        if token.isdigit():
            validated_tokens.append(token)
        # Check if token is an approved operator
        elif token in approved_operators:
            validated_tokens.append(token)
        # Check if token is an approved keyword
        elif token in approved_keywords:
            validated_tokens.append(token)
        # Check if token is a valid identifier (variable/function name)
        elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token) and len(token) <= 100:
            validated_tokens.append(token)
        else:
            raise ValueError(f"Invalid token: '{token}'")
    
    # Verify all characters in script are part of valid tokens or whitespace
    reconstructed = ''
    for token in tokens:
        reconstructed += token
    
    # Remove all whitespace from original script for comparison
    script_no_whitespace = re.sub(r'\s+', '', script)
    reconstructed_no_whitespace = re.sub(r'\s+', '', reconstructed)
    
    if script_no_whitespace != reconstructed_no_whitespace:
        raise ValueError("Script contains invalid tokens")
    
    return validated_tokens
