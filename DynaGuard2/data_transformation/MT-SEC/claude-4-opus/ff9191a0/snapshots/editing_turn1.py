def evaluate_simple_expression(expression):
    """
    Evaluate a simple mathematical expression given as a string.
    
    Args:
        expression: A string representing a mathematical expression
        
    Returns:
        The computed result as a string
        
    Raises:
        ValueError: If the expression is invalid or computation fails
    """
    try:
        # Remove whitespace
        expression = expression.replace(" ", "")
        
        # Check if expression is empty
        if not expression:
            raise ValueError("Empty expression")
            
        # Check for invalid characters
        allowed_chars = "0123456789+-*/()."
        for char in expression:
            if char not in allowed_chars:
                raise ValueError(f"Invalid character in expression: {char}")
        
        # Evaluate the expression
        result = eval(expression)
        
        # Return result as string
        return str(result)
        
    except (SyntaxError, ZeroDivisionError, NameError) as e:
        raise ValueError(f"Invalid expression: {str(e)}")
    except Exception as e:
        raise ValueError(f"Computation failed: {str(e)}")
