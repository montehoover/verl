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
    
    # Verify all characters in script are part of valid tokens or whitespace
    reconstructed = ''
    for token in tokens:
        reconstructed += token
    
    # Remove all whitespace from original script for comparison
    script_no_whitespace = re.sub(r'\s+', '', script)
    reconstructed_no_whitespace = re.sub(r'\s+', '', reconstructed)
    
    if script_no_whitespace != reconstructed_no_whitespace:
        raise ValueError("Script contains invalid tokens")
    
    return tokens
