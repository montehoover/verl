def evaluate_simple_expression(expression, variables=None):
    """
    Evaluate a simple mathematical expression and return the result as a string.
    
    Args:
        expression (str): A string containing a mathematical expression with numbers,
                         operators (+, -, *, /), and optionally variable names
        variables (dict): Optional dictionary mapping variable names to numeric values
    
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
        
        # If variables provided, substitute them
        if variables:
            # Create a copy to avoid modifying the original
            eval_expression = expression
            
            # Sort variables by length (descending) to handle overlapping names correctly
            sorted_vars = sorted(variables.items(), key=lambda x: len(x[0]), reverse=True)
            
            for var_name, var_value in sorted_vars:
                # Validate variable name (alphanumeric and underscore only, not starting with digit)
                if not var_name.replace('_', '').isalnum() or var_name[0].isdigit():
                    raise ValueError(f"Invalid variable name: {var_name}")
                
                # Replace variable with its value in parentheses to preserve precedence
                eval_expression = eval_expression.replace(var_name, f"({var_value})")
        else:
            eval_expression = expression
        
        # Check for invalid characters (now including letters for variables)
        if variables:
            valid_chars = set('0123456789+-*/.()_' + ''.join(variables.keys()))
        else:
            valid_chars = set('0123456789+-*/.)')
            
        if not all(c in valid_chars for c in expression):
            raise ValueError("Invalid characters in expression")
        
        # Check for security issues (no function calls, imports, etc.)
        if any(keyword in eval_expression for keyword in ['import', '__', 'eval', 'exec', 'open', 'file', 'input', 'raw_input']):
            raise ValueError("Invalid expression")
        
        # Evaluate the expression
        result = eval(eval_expression)
        
        # Return as string
        return str(result)
        
    except (SyntaxError, ZeroDivisionError, NameError, TypeError) as e:
        raise ValueError(f"Invalid expression: {str(e)}")
    except Exception as e:
        raise ValueError(f"Cannot process expression: {str(e)}")
