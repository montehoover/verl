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
        # Replace variables with their values
        expr = calc_expr
        for var_name, var_value in var_dict.items():
            # Use word boundaries to avoid partial replacements
            expr = re.sub(r'\b' + re.escape(var_name) + r'\b', str(var_value), expr)
        
        # Parse and evaluate the expression safely using AST
        node = ast.parse(expr, mode='eval')
        
        # Check that the expression only contains allowed operations
        for element in ast.walk(node):
            if isinstance(element, ast.Call):
                raise ValueError("Function calls are not allowed")
            elif isinstance(element, (ast.Import, ast.ImportFrom)):
                raise ValueError("Import statements are not allowed")
            elif isinstance(element, ast.Name):
                raise ValueError(f"Undefined variable: {element.id}")
        
        # Compile and evaluate the expression
        code = compile(node, '<string>', 'eval')
        result = eval(code, {"__builtins__": {}}, {})
        
        # Return the result as a string
        return str(result)
        
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}")
    except Exception as e:
        raise ValueError(f"Computation failed: {e}")
