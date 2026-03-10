import re
import operator
import ast

def execute_calculation_string(calc_expr, var_dict):
    """
    Takes a mathematical expression as a string and computes the simplified result.
    
    Args:
        calc_expr (str): A string representing a mathematical expression
        var_dict (dict): A dictionary where keys are variable names and values are their corresponding numbers
    
    Returns:
        str: The result of the simplified expression as a string
    
    Raises:
        ValueError: If the input expression is invalid or the computation fails
    """
    try:
        # Create a copy of the expression to work with
        expression = calc_expr
        
        # Sort variables by length (descending) to avoid partial replacements
        # e.g., 'x1' should be replaced before 'x'
        sorted_vars = sorted(var_dict.keys(), key=len, reverse=True)
        
        # Replace variables with their values
        for var_name in sorted_vars:
            # Use word boundaries to ensure we replace whole variable names only
            pattern = r'\b' + re.escape(var_name) + r'\b'
            value_str = str(var_dict[var_name])
            
            # Add parentheses around negative numbers to preserve correct order of operations
            if var_dict[var_name] < 0:
                value_str = f'({value_str})'
            
            expression = re.sub(pattern, value_str, expression)
        
        # Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')
        
        # Compile and evaluate the expression safely
        compiled = compile(tree, '<string>', 'eval')
        
        # Create a safe namespace with only basic math operations
        safe_dict = {
            '__builtins__': {},
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
        }
        
        # Evaluate the expression
        result = eval(compiled, safe_dict)
        
        # Convert result to string
        # Handle floating point results by converting to int if they're whole numbers
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        else:
            return str(result)
            
    except SyntaxError:
        raise ValueError(f"Invalid expression syntax: {calc_expr}")
    except ZeroDivisionError:
        raise ValueError("Division by zero in expression")
    except Exception as e:
        raise ValueError(f"Failed to compute expression: {str(e)}")
