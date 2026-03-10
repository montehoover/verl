import re
import operator
import ast

def substitute_variables(expression, variables):
    """
    Substitutes variables in a mathematical expression with their values.
    
    Args:
        expression: A string containing the mathematical expression
        variables: A dictionary mapping variable names to their values
        
    Returns:
        A string with variables substituted by their values
    """
    def replace_var(match):
        var_name = match.group(0)
        if var_name in variables:
            return str(variables[var_name])
        return var_name
    
    # Pattern to match variable names (letters, numbers, underscores)
    # but must start with a letter or underscore
    pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
    
    return re.sub(pattern, replace_var, expression)

def evaluate_expression(expression):
    """
    Evaluates a mathematical expression and returns the result as a string.
    
    Args:
        expression: A string containing the mathematical expression
        
    Returns:
        The computed result as a string
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def execute_calculation_string(calc_expr, var_dict):
    """
    Handles both variable substitution and expression evaluation.
    
    Args:
        calc_expr: A string representing the mathematical expression
        var_dict: A dictionary with variable names and their values
        
    Returns:
        The computed result as a string
        
    Raises:
        ValueError: If the input is invalid or computation fails
    """
    try:
        # First substitute variables
        substituted_expr = substitute_variables(calc_expr, var_dict)
        
        # Validate the expression using AST
        node = ast.parse(substituted_expr, mode='eval')
        
        # Check if the AST contains only allowed operations
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                raise ValueError("Function calls are not allowed")
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                raise ValueError("Import statements are not allowed")
        
        # Evaluate the expression
        result = eval(compile(node, '<string>', 'eval'))
        return str(result)
        
    except (SyntaxError, TypeError, ZeroDivisionError, NameError) as e:
        raise ValueError(f"Invalid expression or computation failed: {str(e)}")
    except Exception as e:
        raise ValueError(f"Computation failed: {str(e)}")
