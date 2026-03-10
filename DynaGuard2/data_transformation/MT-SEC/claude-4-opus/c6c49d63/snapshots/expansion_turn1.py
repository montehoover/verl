import re

def is_valid_expression(expression):
    """
    Validates if a string contains only numbers, basic arithmetic operators (+, -, *, /),
    parentheses, and spaces.
    
    Args:
        expression (str): The expression string to validate
        
    Returns:
        bool: True if the expression is valid, False otherwise
    """
    # Pattern allows:
    # - digits (0-9)
    # - decimal points
    # - arithmetic operators: +, -, *, /
    # - parentheses: ( )
    # - spaces
    pattern = r'^[0-9+\-*/().\s]+$'
    
    if not expression:
        return False
    
    # Check if the expression matches the allowed pattern
    if not re.match(pattern, expression):
        return False
    
    # Additional validation to prevent empty operators or invalid patterns
    # Check for invalid patterns like multiple operators in a row
    invalid_patterns = [
        r'[+\-*/]{2,}',  # Multiple operators in a row (except for negative numbers)
        r'^\s*[*/]',     # Starting with * or /
        r'[+\-*/]\s*$',  # Ending with an operator
        r'\(\s*\)',      # Empty parentheses
        r'[+*/]\s*[+*/]', # Multiple operators (excluding minus for negative numbers)
    ]
    
    for pattern in invalid_patterns:
        if re.search(pattern, expression):
            return False
    
    # Check for balanced parentheses
    open_count = expression.count('(')
    close_count = expression.count(')')
    if open_count != close_count:
        return False
    
    # Check parentheses are properly ordered
    count = 0
    for char in expression:
        if char == '(':
            count += 1
        elif char == ')':
            count -= 1
            if count < 0:
                return False
    
    return True


# Example usage
if __name__ == "__main__":
    # Test cases
    test_expressions = [
        "2 + 3 * 4",           # Valid
        "(10 - 5) / 2",        # Valid
        "3.14 * 2",            # Valid
        "-5 + 10",             # Valid
        "2 ++ 3",              # Invalid - multiple operators
        "import os",           # Invalid - contains letters
        "2 & 3",               # Invalid - unsupported operator
        "((2 + 3)",            # Invalid - unbalanced parentheses
        "",                    # Invalid - empty
        "2 + ",                # Invalid - ends with operator
        "() + 5",              # Invalid - empty parentheses
    ]
    
    for expr in test_expressions:
        result = is_valid_expression(expr)
        print(f"'{expr}' -> {result}")
