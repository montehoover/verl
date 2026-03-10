import re

def validate_expression(expression):
    """
    Validates whether a mathematical expression is safe for evaluation.
    
    Args:
        expression (str): The mathematical expression to validate
        
    Returns:
        bool: True if the expression is valid and safe, False otherwise
    """
    if not expression or not isinstance(expression, str):
        return False
    
    # Remove all whitespace for easier validation
    expr = expression.replace(" ", "")
    
    # Check for empty expression after removing spaces
    if not expr:
        return False
    
    # Define allowed characters: digits, operators, parentheses, and decimal point
    allowed_pattern = r'^[0-9+\-*/().\s]+$'
    
    # Check if expression contains only allowed characters
    if not re.match(allowed_pattern, expression):
        return False
    
    # Check for dangerous patterns
    dangerous_patterns = [
        r'__',  # Double underscores (could be used for special attributes)
        r'import',  # Import statements
        r'exec',  # Exec function
        r'eval',  # Eval function
        r'open',  # File operations
        r'file',  # File operations
        r'input',  # Input function
        r'raw_input',  # Raw input function
        r'\[',  # List brackets
        r'\]',  # List brackets
        r'\{',  # Dict/set brackets
        r'\}',  # Dict/set brackets
        r',',  # Comma (could be used for tuples)
        r';',  # Semicolon
        r':',  # Colon
        r'=',  # Assignment
        r'lambda',  # Lambda functions
        r'def',  # Function definitions
        r'class',  # Class definitions
        r'for',  # Loops
        r'while',  # Loops
        r'if',  # Conditionals
        r'else',  # Conditionals
        r'elif',  # Conditionals
        r'try',  # Exception handling
        r'except',  # Exception handling
        r'raise',  # Exception raising
        r'assert',  # Assertions
        r'del',  # Delete statements
        r'global',  # Global declarations
        r'nonlocal',  # Nonlocal declarations
        r'yield',  # Generators
        r'return',  # Return statements
        r'pass',  # Pass statements
        r'break',  # Break statements
        r'continue',  # Continue statements
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, expression, re.IGNORECASE):
            return False
    
    # Check for consecutive operators (except for negative numbers)
    if re.search(r'[+*/]{2,}', expr):
        return False
    
    # Check for operators at the beginning (except minus for negative numbers)
    if re.match(r'^[+*/]', expr):
        return False
    
    # Check for operators at the end
    if re.search(r'[+\-*/]$', expr):
        return False
    
    # Check for empty parentheses
    if '()' in expr:
        return False
    
    # Check for balanced parentheses
    paren_count = 0
    for char in expr:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
            if paren_count < 0:
                return False
    
    if paren_count != 0:
        return False
    
    # Check for valid decimal number format
    # Split by operators and parentheses to get individual terms
    terms = re.split(r'[+\-*/()]', expr)
    for term in terms:
        if term:  # Skip empty strings from split
            # Check if term is a valid number
            if not re.match(r'^\d+\.?\d*$', term):
                return False
            # Check for multiple decimal points
            if term.count('.') > 1:
                return False
    
    # If all checks pass, the expression is valid
    return True


def evaluate_safe_expression(expression):
    """
    Evaluates a mathematical expression after validation.
    
    Args:
        expression (str): The mathematical expression to evaluate
        
    Returns:
        float or str: The result of the evaluation, or an error message
    """
    # First validate the expression
    if not validate_expression(expression):
        return "Error: Invalid expression"
    
    try:
        # Use eval with restricted globals and locals
        # Only allow basic math operations
        allowed_names = {
            '__builtins__': None,
            'abs': abs,
            'min': min,
            'max': max,
        }
        
        # Evaluate the expression
        result = eval(expression, {"__builtins__": None}, {})
        
        # Check for division by zero or other special cases
        if isinstance(result, float):
            if result == float('inf') or result == float('-inf'):
                return "Error: Division by zero"
            if result != result:  # Check for NaN
                return "Error: Invalid computation"
        
        return float(result)
        
    except ZeroDivisionError:
        return "Error: Division by zero"
    except (ValueError, TypeError):
        return "Error: Invalid computation"
    except Exception as e:
        return "Error: Computation problem"


def calculate_expression(expr_input):
    """
    Processes and evaluates user-submitted arithmetic expressions securely.
    
    Args:
        expr_input (str): The arithmetic expression to calculate
        
    Returns:
        float or str: The result of the calculation, or 'Computation Error!' if invalid
    """
    # Evaluate the expression using our safe evaluation function
    result = evaluate_safe_expression(expr_input)
    
    # Check if the result is an error message
    if isinstance(result, str) and result.startswith("Error:"):
        return "Computation Error!"
    
    return result


# Example usage and tests
if __name__ == "__main__":
    # Valid expressions
    valid_expressions = [
        "2 + 3",
        "10 - 5",
        "4 * 6",
        "15 / 3",
        "(2 + 3) * 4",
        "3.14 * 2",
        "100 - (20 + 30)",
        "-5 + 10",
        "(-3) * 4",
        "2.5 * 4.0",
    ]
    
    # Invalid expressions
    invalid_expressions = [
        "__import__('os').system('ls')",
        "eval('malicious code')",
        "2 + 3; print('hack')",
        "[1, 2, 3]",
        "{'key': 'value'}",
        "lambda x: x + 1",
        "2 ** 3",  # Exponentiation not allowed
        "import os",
        "2++3",
        "++3",
        "*5",
        "5-",
        "()",
        "((2 + 3)",
        "2 + 3)",
        "2..3",
        "",
        "   ",
        None,
    ]
    
    print("Testing valid expressions:")
    for expr in valid_expressions:
        result = validate_expression(expr)
        print(f"'{expr}' -> {result}")
    
    print("\nTesting invalid expressions:")
    for expr in invalid_expressions:
        result = validate_expression(expr)
        print(f"'{expr}' -> {result}")
    
    print("\n" + "="*50 + "\n")
    
    # Test evaluate_safe_expression
    test_expressions = [
        "2 + 3",
        "10 - 5",
        "4 * 6",
        "15 / 3",
        "(2 + 3) * 4",
        "3.14 * 2",
        "100 - (20 + 30)",
        "-5 + 10",
        "(-3) * 4",
        "2.5 * 4.0",
        "10 / 0",  # Division by zero
        "1 / 0.0",  # Division by zero (float)
        "__import__('os')",  # Should be caught by validation
        "2 ** 3",  # Not allowed
        "invalid",  # Invalid expression
    ]
    
    print("Testing evaluate_safe_expression:")
    for expr in test_expressions:
        result = evaluate_safe_expression(expr)
        print(f"'{expr}' -> {result}")
    
    print("\n" + "="*50 + "\n")
    
    # Test calculate_expression
    print("Testing calculate_expression:")
    test_calc_expressions = [
        "2 + 3",
        "10 - 5",
        "4 * 6",
        "15 / 3",
        "(2 + 3) * 4",
        "3.14 * 2",
        "100 - (20 + 30)",
        "-5 + 10",
        "(-3) * 4",
        "2.5 * 4.0",
        "10 / 0",  # Should return 'Computation Error!'
        "1 / 0.0",  # Should return 'Computation Error!'
        "__import__('os')",  # Should return 'Computation Error!'
        "2 ** 3",  # Should return 'Computation Error!'
        "invalid",  # Should return 'Computation Error!'
        "",  # Should return 'Computation Error!'
        "2++3",  # Should return 'Computation Error!'
    ]
    
    for expr in test_calc_expressions:
        result = calculate_expression(expr)
        print(f"'{expr}' -> {result}")
