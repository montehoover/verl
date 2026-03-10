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

def parse_expression(expression):
    """
    Parses a valid expression string and returns a list of numbers and operators.
    
    Args:
        expression (str): The expression string to parse
        
    Returns:
        list: A list containing numbers (as floats) and operators (as strings)
    """
    # Remove all spaces for easier parsing
    expression = expression.replace(' ', '')
    
    # Pattern to match numbers (including decimals and negative numbers) and operators
    # This pattern captures:
    # - Numbers (including decimals): \d+\.?\d*
    # - Operators and parentheses: [+\-*/()]
    pattern = r'(-?\d+\.?\d*|[+\-*/()])'
    
    tokens = re.findall(pattern, expression)
    
    result = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # Check if it's a number
        if re.match(r'-?\d+\.?\d*$', token):
            result.append(float(token))
        # Check for negative number (minus followed by a number)
        elif token == '-' and i + 1 < len(tokens) and re.match(r'\d+\.?\d*$', tokens[i + 1]):
            # Check if this minus is for a negative number or subtraction
            if i == 0 or tokens[i - 1] in ['(', '+', '-', '*', '/']:
                # It's a negative number
                result.append(-float(tokens[i + 1]))
                i += 1  # Skip the next token as we've already processed it
            else:
                # It's a subtraction operator
                result.append(token)
        else:
            # It's an operator or parenthesis
            result.append(token)
        
        i += 1
    
    return result

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
    
    print("Validation tests:")
    for test in test_cases:
        result = validate_expression(test)
        print(f"'{test}' -> {result}")
    
    print("\nParsing tests:")
    valid_expressions = [
        "2 + 2",
        "10 - 5 * 2",
        "(3 + 4) * 2",
        "3.14 * 2",
        "2 + 3.14159",
        "-5 + 3",
        "10 / 2",
        "(-3 + 5) * 2",
        "-10 * -5"
    ]
    
    for expr in valid_expressions:
        if validate_expression(expr):
            parsed = parse_expression(expr)
            print(f"'{expr}' -> {parsed}")
