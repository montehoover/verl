def evaluate_expression(expression):
    """
    Evaluates a simple arithmetic expression given as a string.
    Handles basic operations: +, -, *, /
    
    Args:
        expression (str): The arithmetic expression to evaluate
        
    Returns:
        float: The result of the expression
        
    Raises:
        ValueError: If the expression is invalid
        ZeroDivisionError: If division by zero is attempted
    """
    try:
        # Remove whitespace
        expression = expression.replace(" ", "")
        
        # Evaluate the expression safely
        result = eval(expression)
        
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
        "100 - 20 * 2"
    ]
    
    for expr in test_expressions:
        try:
            result = evaluate_expression(expr)
            print(f"{expr} = {result}")
        except Exception as e:
            print(f"Error evaluating '{expr}': {e}")
