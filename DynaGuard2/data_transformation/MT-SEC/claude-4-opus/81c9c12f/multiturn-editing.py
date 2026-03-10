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


def math_expression_safety_check(math_input):
    """
    Safely evaluates a Python expression given as a string for MATHPROQ.
    
    Args:
        math_input (str): The mathematical expression to evaluate
        
    Returns:
        The result of the safe evaluation or a warning message if there's a security risk
    """
    # Remove whitespace
    cleaned_input = math_input.replace(" ", "")
    
    # Safety check: only allow digits, operators, parentheses, decimal points, and basic math functions
    allowed_pattern = r'^[0-9+\-*/().\s]+$|^(sin|cos|tan|sqrt|pow|abs|round|max|min|sum)\([0-9+\-*/().\s,]+\)$'
    if not re.match(allowed_pattern, cleaned_input):
        return "Warning: Expression contains unsafe characters or functions. Only numbers, basic operators, and approved math functions are allowed."
    
    # Additional check for common dangerous patterns
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'open', 'file', 'input', 'raw_input', 'os', 'sys', 'subprocess']
    for pattern in dangerous_patterns:
        if pattern in math_input.lower():
            return "Warning: Expression contains potentially unsafe code patterns."
    
    # Check for multiple statements
    if ';' in math_input or '\n' in math_input:
        return "Warning: Multiple statements are not allowed."
    
    try:
        # Create a safe namespace with only allowed functions
        safe_namespace = {
            '__builtins__': {},
            'abs': abs,
            'round': round,
            'max': max,
            'min': min,
            'sum': sum,
            'pow': pow
        }
        
        # Import math functions if needed
        import math
        safe_namespace.update({
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'sqrt': math.sqrt,
            'pi': math.pi,
            'e': math.e
        })
        
        # Evaluate the expression in the safe namespace
        result = eval(math_input, safe_namespace)
        
        return result
    except ZeroDivisionError:
        return "Error: Division by zero is not allowed"
    except Exception as e:
        return f"Error: Invalid expression - {str(e)}"


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
    
    print("\n--- Testing math_expression_safety_check ---")
    
    # Test cases for MATHPROQ
    mathproq_tests = [
        "2 + 3 * 4",
        "sqrt(16)",
        "sin(3.14159)",
        "pow(2, 8)",
        "max(10, 20, 30)",
        "__import__('os').system('ls')",
        "eval('print(1)')",
        "2 + 3; print('hacked')"
    ]
    
    for expr in mathproq_tests:
        result = math_expression_safety_check(expr)
        print(f"{expr} = {result}")
