import re

def validate_math_expression(expression):
    """
    Validates a mathematical expression to ensure it only contains
    numbers and basic arithmetic operators (+, -, *, /, %, //, **).
    Also checks for balanced parentheses.
    
    Args:
        expression (str): The mathematical expression to validate
        
    Returns:
        bool: True if the expression is valid, False otherwise
    """
    # Remove all whitespace for easier processing
    expression = expression.replace(" ", "")
    
    # Check if expression is empty
    if not expression:
        return False
    
    # Define allowed characters: digits, operators, parentheses, and decimal points
    allowed_pattern = r'^[0-9+\-*/%(). ]+$'
    
    # Check if expression contains only allowed characters
    if not re.match(allowed_pattern, expression):
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
    
    # Check for invalid patterns
    invalid_patterns = [
        r'[+\-*/]{2,}',  # Multiple operators in a row (except --)
        r'^[*/]',        # Expression starting with * or /
        r'[+\-*/]$',     # Expression ending with an operator
        r'\(\)',         # Empty parentheses
        r'[+\-*/]\)',    # Operator before closing parenthesis
        r'\([+*/]',      # Opening parenthesis followed by operator (except -)
        r'\)\(',         # Adjacent parentheses without operator
        r'\d\(',         # Number directly before opening parenthesis
        r'\)\d',         # Closing parenthesis directly before number
    ]
    
    for pattern in invalid_patterns:
        if re.search(pattern, expression):
            return False
    
    # Additional check for ** (power operator) - allow it
    expression_check = expression.replace('**', 'P')  # Temporarily replace ** to avoid false positive
    if re.search(r'[+\-*/]{2,}', expression_check):
        return False
    
    # Check for // (floor division) - allow it
    expression_check = expression.replace('//', 'F')  # Temporarily replace // to avoid false positive
    if re.search(r'[+\-*/]{2,}', expression_check):
        return False
    
    return True


# Example usage and test cases
if __name__ == "__main__":
    test_expressions = [
        ("2 + 3", True),
        ("10 * (5 - 3)", True),
        ("3.14 * 2", True),
        ("2 ** 3", True),
        ("10 // 3", True),
        ("5 + + 3", False),
        ("(2 + 3", False),
        ("2 + 3)", False),
        ("2 +* 3", False),
        ("", False),
        ("2 + abc", False),
        ("2 + 3; print('hack')", False),
        ("__import__('os')", False),
        ("2(3)", False),
        ("(2)3", False),
        ("()", False),
        ("2 + ", False),
        ("* 5", False),
    ]
    
    for expr, expected in test_expressions:
        result = validate_math_expression(expr)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{expr}' -> {result} (expected {expected})")
