import re

def is_valid_expression(expression):
    """
    Check if a string is a valid arithmetic expression containing only:
    - Numbers (integers and decimals)
    - Operators: +, -, *, /
    - Parentheses: ( and )
    
    Returns True if valid, False otherwise.
    """
    # Remove all whitespace for easier validation
    expression = expression.replace(" ", "")
    
    # Empty string is not valid
    if not expression:
        return False
    
    # Check for valid characters only
    valid_pattern = r'^[0-9+\-*/().\s]+$'
    if not re.match(valid_pattern, expression):
        return False
    
    # Check for balanced parentheses
    paren_count = 0
    for char in expression:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        if paren_count < 0:
            return False
    if paren_count != 0:
        return False
    
    # Check for valid number format (no multiple dots)
    if '..' in expression:
        return False
    
    # Check for invalid operator sequences
    invalid_sequences = ['++', '--', '**', '//', '+-', '-+', '*+', '/+', 
                        '+*', '+/', '-*', '-/', '*/+', '*/-', '/*', '//', 
                        '(*', '(/+', '(/', '(*', '+)', '-)', '*)', '/)',
                        ')(', '.(', ').', '..']
    for seq in invalid_sequences:
        if seq in expression:
            return False
    
    # Check that expression doesn't start or end with operators
    if expression[0] in '*/':
        return False
    if expression[-1] in '+-*/':
        return False
    
    # Check for empty parentheses
    if '()' in expression:
        return False
    
    # Additional validation using tokenization
    try:
        # Split into tokens
        tokens = re.findall(r'\d+\.?\d*|[+\-*/()]', expression)
        
        # Reconstruct and compare
        reconstructed = ''.join(tokens)
        if reconstructed != expression:
            return False
        
        # Check for proper alternation between numbers and operators
        prev_was_number = False
        prev_was_operator = False
        
        for i, token in enumerate(tokens):
            if re.match(r'\d+\.?\d*', token):  # Number
                if prev_was_number:
                    return False
                prev_was_number = True
                prev_was_operator = False
            elif token in '+-*/':  # Operator
                # Allow unary minus at start or after opening parenthesis
                if token == '-' and (i == 0 or tokens[i-1] == '('):
                    continue
                if prev_was_operator:
                    return False
                prev_was_operator = True
                prev_was_number = False
            elif token in '()':  # Parenthesis
                prev_was_number = False
                prev_was_operator = False
        
        return True
        
    except:
        return False
