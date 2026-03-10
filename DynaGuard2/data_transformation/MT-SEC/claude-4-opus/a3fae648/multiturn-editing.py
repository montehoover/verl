import re
import operator
import ast

def simplify_math_expression(formula_str, vars_mapping):
    """
    Compute and simplify a mathematical expression with variables.
    
    Args:
        formula_str (str): A string representing a mathematical formula with variables
        vars_mapping (dict): A dictionary mapping variable names to values
        
    Returns:
        str: The computed result as a string
        
    Raises:
        ValueError: If the expression is invalid or cannot be processed
    """
    try:
        # Remove whitespace
        formula_str = formula_str.strip()
        
        # Check if expression is empty
        if not formula_str:
            raise ValueError("Empty expression")
        
        # Validate vars_mapping
        if not isinstance(vars_mapping, dict):
            raise ValueError("vars_mapping must be a dictionary")
        
        # Check all variable values are numeric
        for key, value in vars_mapping.items():
            if not isinstance(key, str):
                raise ValueError(f"Variable name must be a string: {key}")
            if not isinstance(value, (int, float)):
                raise ValueError(f"Variable value must be numeric: {key}={value}")
        
        # Parse the expression using AST
        try:
            tree = ast.parse(formula_str, mode='eval')
        except SyntaxError as e:
            raise ValueError(f"Invalid syntax in expression: {e}")
        
        # Define allowed AST node types for safe evaluation
        allowed_nodes = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
            ast.USub, ast.UAdd,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
            ast.Name, ast.Load, ast.Constant, ast.Num
        )
        
        # Validate that the AST only contains allowed operations
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                raise ValueError(f"Unsafe operation in expression: {type(node).__name__}")
        
        # Create a safe namespace with only the provided variables
        safe_namespace = vars_mapping.copy()
        
        # Compile and evaluate the expression
        compiled = compile(tree, '<string>', 'eval')
        result = eval(compiled, {"__builtins__": {}}, safe_namespace)
        
        # Return result as string
        return str(result)
        
    except (ValueError, TypeError) as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Unexpected error: {str(e)}")
