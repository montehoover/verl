def evaluate_expression(expression):
    """
    Evaluate a simple arithmetic expression given as a string.
    
    Args:
        expression (str): The arithmetic expression to evaluate
        
    Returns:
        float or str: The result of the expression or an error message
    """
    try:
        # Remove whitespace
        expression = expression.replace(" ", "")
        
        # Check for empty expression
        if not expression:
            return "Error: Empty expression"
        
        # Check for invalid characters
        allowed_chars = "0123456789+-*/()."
        for char in expression:
            if char not in allowed_chars:
                return f"Error: Invalid character '{char}' in expression"
        
        # Check for division by zero by evaluating
        result = eval(expression)
        
        return result
        
    except ZeroDivisionError:
        return "Error: Division by zero"
    except SyntaxError:
        return "Error: Invalid expression syntax"
    except Exception as e:
        return f"Error: {str(e)}"


# Test the function
if __name__ == "__main__":
    # Test cases
    test_expressions = [
        "2 + 3",
        "10 - 5",
        "4 * 6",
        "15 / 3",
        "2 + 3 * 4",
        "(2 + 3) * 4",
        "10 / 0",
        "2 + + 3",
        "hello",
        "",
        "3.14 * 2",
        "100 / 4 - 5"
    ]
    
    for expr in test_expressions:
        result = evaluate_expression(expr)
        print(f"{expr} = {result}")
