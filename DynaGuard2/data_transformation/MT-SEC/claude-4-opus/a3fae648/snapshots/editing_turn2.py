def evaluate_expression(expression, variables=None):
    """
    Evaluate a simple mathematical expression and return the result as a string.
    
    Args:
        expression (str): A string containing a mathematical expression
        variables (dict): Optional dictionary mapping variable names to numeric values
        
    Returns:
        str: The calculated result as a string
        
    Raises:
        ValueError: If the expression is invalid or cannot be processed
    """
    try:
        # Remove whitespace
        expression = expression.strip()
        
        # Check if expression is empty
        if not expression:
            raise ValueError("Empty expression")
        
        # If variables provided, check they are valid
        if variables is not None:
            if not isinstance(variables, dict):
                raise ValueError("Variables must be provided as a dictionary")
            
            # Check all variable values are numeric
            for key, value in variables.items():
                if not isinstance(key, str):
                    raise ValueError(f"Variable name must be a string: {key}")
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Variable value must be numeric: {key}={value}")
        
        # Check for invalid characters (now including letters for variables)
        allowed_chars = set('0123456789+-*/.()')
        if variables:
            # Add allowed variable name characters
            allowed_chars.update('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_')
        
        if not all(c in allowed_chars or c.isspace() for c in expression):
            raise ValueError("Invalid characters in expression")
        
        # Create evaluation namespace
        eval_namespace = {}
        if variables:
            eval_namespace.update(variables)
        
        # Evaluate the expression
        result = eval(expression, {"__builtins__": {}}, eval_namespace)
        
        # Return result as string
        return str(result)
        
    except (SyntaxError, ZeroDivisionError, NameError, TypeError) as e:
        raise ValueError(f"Invalid expression: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")
