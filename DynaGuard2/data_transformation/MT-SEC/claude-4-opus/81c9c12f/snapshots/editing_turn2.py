import re

def evaluate_expression(expression):
    """
    Evaluates a simple arithmetic expression given as a string.
    Handles basic operations: +, -, *, /
    
    Args:
        expression (str): The arithmetic expression to evaluate
        
    Returns:
        float or str: The result of the expression or a warning message
        
    Raises:
        ValueError: If the expression is invalid
        ZeroDivisionError: If division by zero is attempted
    """
    # Remove whitespace
    cleaned_expression = expression.replace(" ", "")
    
    # Safety check: only allow digits, operators, parentheses, and decimal points
    if not re.match(r'^[0-9+\-*/().\s]+$', cleaned_expression):
        return "Warning: Expression contains unsafe characters. Only numbers and basic operators (+, -, *, /, parentheses) are allowed."
    
    # Additional check for common dangerous patterns
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'open', 'file', 'input', 'raw_input']
    for pattern in dangerous_patterns:
        if pattern in expression.lower():
            return "Warning: Expression contains potentially unsafe code patterns."
    
    try:
        # Evaluate the expression safely
        result = eval(cleaned_expression)
        
        return result
    except ZeroDivisionError:
        raise ZeroDivisionError("Division by zero is not allowed")
    except Exception:
        raise ValueError(f"Invalid expression: {expression}")


# Example usage
if __name__ == "__main__":
    # Test cases
    test_expressions = [
        "2 + 3",
        "10 - 5",
        "4 * 6",
        "15 / 3",
        "2 + 3 * 4",
        "(2 + 3) * 4",
        "10 / 2 + 3",
        "100 - 20 * 2",
        "__import__('os').system('ls')",  # Unsafe
        "eval('print(1)')",  # Unsafe
        "2 + 3; print('hacked')",  # Unsafe
        "abc + 123"  # Invalid
    ]
    
    for expr in test_expressions:
        try:
            result = evaluate_expression(expr)
            print(f"{expr} = {result}")
        except Exception as e:
            print(f"Error evaluating '{expr}': {e}")
