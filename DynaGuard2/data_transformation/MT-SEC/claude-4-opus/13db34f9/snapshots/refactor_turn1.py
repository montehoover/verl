import re
import operator
import ast

def evaluate_math_expression(formula, vars):
    """
    Process a mathematical expression string with variables and return the calculated result.
    
    Args:
        formula (str): A string representing a mathematical formula potentially containing variables
        vars (dict): A mapping of variable names to their numeric values for evaluation
        
    Returns:
        str: The result after computing the expression, returned in string format
        
    Raises:
        ValueError: If an error occurs due to an invalid expression or unsuccessful processing
    """
    try:
        # Replace variables in the formula with their values
        processed_formula = formula
        for var_name, var_value in vars.items():
            # Use word boundaries to avoid partial replacements
            pattern = r'\b' + re.escape(var_name) + r'\b'
            processed_formula = re.sub(pattern, str(var_value), processed_formula)
        
        # Parse and evaluate the expression using ast
        node = ast.parse(processed_formula, mode='eval')
        
        # Check if the parsed expression is safe (only contains allowed operations)
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                # Check for any remaining unsubstituted variables
                raise ValueError(f"Undefined variable: {n.id}")
            elif not isinstance(n, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, 
                                  ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, 
                                  ast.Pow, ast.USub, ast.UAdd, ast.Mod, ast.FloorDiv)):
                raise ValueError("Invalid operation in expression")
        
        # Evaluate the expression
        result = eval(compile(node, '<string>', 'eval'))
        
        # Convert result to string and return
        return str(result)
        
    except (SyntaxError, TypeError, ZeroDivisionError, NameError) as e:
        raise ValueError(f"Error processing expression: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error: {str(e)}")
