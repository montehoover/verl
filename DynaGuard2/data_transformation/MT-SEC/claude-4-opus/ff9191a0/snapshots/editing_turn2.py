def evaluate_simple_expression(expression, variables=None):
    """
    Evaluate a simple mathematical expression given as a string.
    
    Args:
        expression: A string representing a mathematical expression
        variables: A dictionary mapping variable names to their numeric values
        
    Returns:
        The computed result as a string
        
    Raises:
        ValueError: If the expression is invalid, computation fails, or a variable is missing
    """
    try:
        # Remove whitespace
        expression = expression.replace(" ", "")
        
        # Check if expression is empty
        if not expression:
            raise ValueError("Empty expression")
        
        # If no variables provided, create empty dict
        if variables is None:
            variables = {}
            
        # Check for invalid characters (now including letters for variables)
        allowed_chars = "0123456789+-*/().abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"
        for char in expression:
            if char not in allowed_chars:
                raise ValueError(f"Invalid character in expression: {char}")
        
        # Create a safe namespace with only the provided variables
        safe_dict = {}
        for var_name, var_value in variables.items():
            if not isinstance(var_value, (int, float)):
                raise ValueError(f"Variable '{var_name}' must be numeric")
            safe_dict[var_name] = var_value
        
        # Evaluate the expression with the safe namespace
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        
        # Return result as string
        return str(result)
        
    except NameError as e:
        # Extract variable name from error message
        var_name = str(e).split("'")[1]
        raise ValueError(f"Variable '{var_name}' is not defined in the provided mapping")
    except (SyntaxError, ZeroDivisionError) as e:
        raise ValueError(f"Invalid expression: {str(e)}")
    except Exception as e:
        raise ValueError(f"Computation failed: {str(e)}")
