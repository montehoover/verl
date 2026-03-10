import re

def validate_expression(expression):
    """
    Validates if a string contains only numbers, basic arithmetic operators (+, -, *, /), 
    parentheses, decimal points, and spaces.
    
    Args:
        expression (str): The expression string to validate
        
    Returns:
        bool: True if the expression is valid, False otherwise
    """
    # Pattern allows:
    # - digits (0-9)
    # - decimal points
    # - arithmetic operators (+, -, *, /)
    # - parentheses for grouping
    # - spaces
    pattern = r'^[0-9+\-*/().\s]+$'
    
    if not expression or not expression.strip():
        return False
    
    if not re.match(pattern, expression):
        return False
    
    # Additional validation to prevent invalid patterns
    # Check for empty parentheses
    if '()' in expression:
        return False
    
    # Check for multiple consecutive operators (except for negative numbers)
    if re.search(r'[+*/]{2,}', expression):
        return False
    
    # Check for operators at the end
    if re.search(r'[+\-*/]\s*$', expression):
        return False
    
    # Check for operators at the beginning (except minus for negative numbers)
    if re.search(r'^\s*[+*/]', expression):
        return False
    
    return True

# Example usage
if __name__ == "__main__":
    test_cases = [
        "2 + 2",
        "10 - 5 * 2",
        "(3 + 4) * 2",
        "3.14 * 2",
        "2 + 2; print('hack')",
        "2 ** 3",
        "2 + ",
        " * 5",
        "2 ++ 3",
        "()",
        "2 + 3.14159",
        "-5 + 3",
        "10 / 2"
    ]
    
    for test in test_cases:
        result = validate_expression(test)
        print(f"'{test}' -> {result}")
