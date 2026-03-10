import re
import ast

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


def apply_operator(num1, operator, num2):
    """
    Applies an operator to two numbers and returns the result.
    
    Args:
        num1 (float): First number
        operator (str): Operator (+, -, *, /)
        num2 (float): Second number
        
    Returns:
        float: Result of the operation
        
    Raises:
        ValueError: If operator is not supported
        ZeroDivisionError: If dividing by zero
    """
    if operator == '+':
        return num1 + num2
    elif operator == '-':
        return num1 - num2
    elif operator == '*':
        return num1 * num2
    elif operator == '/':
        if num2 == 0:
            raise ZeroDivisionError("Division by zero")
        return num1 / num2
    else:
        raise ValueError(f"Unsupported operator: {operator}")


def evaluate_expression(tokens):
    """
    Evaluates a list of numbers and operators respecting operator precedence.
    
    Args:
        tokens (list): List of numbers and operators
        
    Returns:
        float: Result of the expression
        
    Raises:
        ValueError: If unsupported operators are found
    """
    if not tokens:
        raise ValueError("Empty expression")
    
    # Convert string numbers to floats
    processed_tokens = []
    for token in tokens:
        if isinstance(token, (int, float)):
            processed_tokens.append(float(token))
        elif isinstance(token, str):
            try:
                processed_tokens.append(float(token))
            except ValueError:
                if token in ['+', '-', '*', '/']:
                    processed_tokens.append(token)
                else:
                    raise ValueError(f"Invalid token: {token}")
        else:
            raise ValueError(f"Invalid token type: {type(token)}")
    
    # First pass: handle multiplication and division
    i = 0
    while i < len(processed_tokens):
        if i + 2 < len(processed_tokens) and processed_tokens[i + 1] in ['*', '/']:
            num1 = processed_tokens[i]
            operator = processed_tokens[i + 1]
            num2 = processed_tokens[i + 2]
            
            result = apply_operator(num1, operator, num2)
            
            # Replace the three tokens with the result
            processed_tokens = processed_tokens[:i] + [result] + processed_tokens[i + 3:]
        else:
            i += 1
    
    # Second pass: handle addition and subtraction
    i = 0
    while i < len(processed_tokens):
        if i + 2 < len(processed_tokens) and processed_tokens[i + 1] in ['+', '-']:
            num1 = processed_tokens[i]
            operator = processed_tokens[i + 1]
            num2 = processed_tokens[i + 2]
            
            result = apply_operator(num1, operator, num2)
            
            # Replace the three tokens with the result
            processed_tokens = processed_tokens[:i] + [result] + processed_tokens[i + 3:]
        else:
            i += 1
    
    # Should have only one token left
    if len(processed_tokens) != 1:
        raise ValueError("Invalid expression structure")
    
    return processed_tokens[0]


def compute_expression(input_expr):
    """
    Computes the result of a mathematical expression string.
    
    Args:
        input_expr (str): The expression string to compute
        
    Returns:
        float: The computed result
        
    Raises:
        ValueError: If the input expression is invalid
    """
    # Validate the expression
    if not is_valid_expression(input_expr):
        raise ValueError("Invalid expression")
    
    # Use ast.literal_eval to safely evaluate the expression
    try:
        result = ast.literal_eval(input_expr)
        return float(result)
    except:
        # If ast.literal_eval fails, parse and evaluate manually
        # Tokenize the expression
        tokens = []
        current_token = ""
        
        for char in input_expr:
            if char in "+-*/":
                if current_token:
                    tokens.append(current_token.strip())
                    current_token = ""
                tokens.append(char)
            elif char == " ":
                continue
            else:
                current_token += char
        
        if current_token:
            tokens.append(current_token.strip())
        
        # Handle parentheses by evaluating inner expressions first
        while '(' in input_expr:
            # Find the innermost parentheses
            start = -1
            for i, char in enumerate(input_expr):
                if char == '(':
                    start = i
                elif char == ')' and start != -1:
                    # Extract and evaluate the inner expression
                    inner_expr = input_expr[start+1:i]
                    inner_result = compute_expression(inner_expr)
                    
                    # Replace the parentheses expression with its result
                    input_expr = input_expr[:start] + str(inner_result) + input_expr[i+1:]
                    break
            else:
                raise ValueError("Unmatched parentheses")
        
        # Re-tokenize after parentheses evaluation
        tokens = []
        current_token = ""
        
        for char in input_expr:
            if char in "+-*/":
                if current_token:
                    tokens.append(current_token.strip())
                    current_token = ""
                tokens.append(char)
            elif char == " ":
                continue
            else:
                current_token += char
        
        if current_token:
            tokens.append(current_token.strip())
        
        # Evaluate the expression
        return evaluate_expression(tokens)


# Example usage
if __name__ == "__main__":
    # Test cases for is_valid_expression
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
    
    print("Validation tests:")
    for expr in test_expressions:
        result = is_valid_expression(expr)
        print(f"'{expr}' -> {result}")
    
    # Test cases for evaluate_expression
    print("\nEvaluation tests:")
    test_evaluations = [
        ([2, '+', 3, '*', 4], 14),           # 2 + 3 * 4 = 2 + 12 = 14
        ([10, '-', 5, '/', 2], 7.5),         # 10 - 5 / 2 = 10 - 2.5 = 7.5
        ([3.14, '*', 2], 6.28),              # 3.14 * 2 = 6.28
        ([8, '/', 2, '/', 2], 2.0),          # 8 / 2 / 2 = 4 / 2 = 2
        (['2', '+', '3'], 5),                # String numbers
    ]
    
    for tokens, expected in test_evaluations:
        try:
            result = evaluate_expression(tokens)
            print(f"{tokens} = {result} (expected: {expected})")
        except Exception as e:
            print(f"{tokens} -> Error: {e}")
    
    # Test cases for compute_expression
    print("\nCompute expression tests:")
    test_computations = [
        ("2 + 3 * 4", 14),
        ("(10 - 5) / 2", 2.5),
        ("3.14 * 2", 6.28),
        ("8 / 2 / 2", 2.0),
        ("(2 + 3) * 4", 20),
        ("10 + (5 * 2)", 20),
    ]
    
    for expr, expected in test_computations:
        try:
            result = compute_expression(expr)
            print(f"'{expr}' = {result} (expected: {expected})")
        except Exception as e:
            print(f"'{expr}' -> Error: {e}")
