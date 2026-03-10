def evaluate_simple_expression(expression):
    """
    Evaluate a simple mathematical expression and return the result as a string.
    
    Args:
        expression (str): A string containing a mathematical expression with numbers
                         and operators (+, -, *, /)
    
    Returns:
        str: The calculated result as a string
        
    Raises:
        ValueError: If the expression is invalid or cannot be processed
    """
    try:
        # Remove whitespace
        expression = expression.replace(" ", "")
        
        # Check if expression is empty
        if not expression:
            raise ValueError("Empty expression")
        
        # Check for invalid characters
        valid_chars = set('0123456789+-*/.)')
        if not all(c in valid_chars for c in expression):
            raise ValueError("Invalid characters in expression")
        
        # Check for security issues (no function calls, imports, etc.)
        if any(keyword in expression for keyword in ['import', '__', 'eval', 'exec', 'open', 'file', 'input', 'raw_input']):
            raise ValueError("Invalid expression")
        
        # Evaluate the expression
        result = eval(expression)
        
        # Return as string
        return str(result)
        
    except (SyntaxError, ZeroDivisionError, NameError, TypeError) as e:
        raise ValueError(f"Invalid expression: {str(e)}")
    except Exception as e:
        raise ValueError(f"Cannot process expression: {str(e)}")
