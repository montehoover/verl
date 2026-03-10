import re
import operator
import ast

def evaluate_math_expression(formula, vars):
    """
    Process a mathematical expression string with variables and return the simplified result as a string.
    
    Args:
        formula (str): A string containing a mathematical expression
        vars (dict): Dictionary mapping variable names to values
    
    Returns:
        str: The calculated result as a string
        
    Raises:
        ValueError: If the expression is invalid or cannot be processed
    """
    try:
        # Replace variables with their values
        expression = formula
        if vars:
            # Sort by length to handle overlapping variable names
            sorted_vars = sorted(vars.items(), key=lambda x: len(x[0]), reverse=True)
            for var_name, var_value in sorted_vars:
                # Use word boundaries to match whole variable names only
                pattern = r'\b' + re.escape(var_name) + r'\b'
                expression = re.sub(pattern, str(var_value), expression)
        
        # Parse the expression using ast to ensure it's safe
        tree = ast.parse(expression, mode='eval')
        
        # Check that the expression only contains allowed operations
        allowed_nodes = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Num,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd
        )
        
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                # Handle ast.Name nodes for Python 3.7 compatibility
                if isinstance(node, ast.Name):
                    raise ValueError(f"Undefined variable: {node.id}")
                raise ValueError("Invalid expression")
        
        # Compile and evaluate the expression
        code = compile(tree, '<string>', 'eval')
        result = eval(code)
        
        # Return result as string
        return str(result)
        
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except SyntaxError:
        raise ValueError("Invalid syntax in expression")
    except Exception as e:
        raise ValueError(f"Cannot process expression: {str(e)}")
