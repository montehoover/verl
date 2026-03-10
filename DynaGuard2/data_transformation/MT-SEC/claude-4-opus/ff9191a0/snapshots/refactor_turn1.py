import re
import operator
import ast

def evaluate_expression_safely(math_expression, variable_mapping):
    """
    Evaluate a mathematical expression with variable substitution.
    
    Args:
        math_expression (str): A string representing a mathematical expression
        variable_mapping (dict): A dictionary where keys are variable names and values are their corresponding numbers
    
    Returns:
        str: The result of the simplified expression as a string
    
    Raises:
        ValueError: If the input expression is invalid or the computation fails
    """
    try:
        # Substitute variables in the expression
        expression = math_expression
        for var, value in variable_mapping.items():
            # Use word boundaries to avoid partial matches
            expression = re.sub(r'\b' + re.escape(var) + r'\b', str(value), expression)
        
        # Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')
        
        # Check if the AST only contains safe operations
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                # Check if there are any remaining variables
                raise ValueError(f"Undefined variable: {node.id}")
            elif not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, 
                                     ast.Add, ast.Sub, ast.Mult, ast.Div, 
                                     ast.Pow, ast.Mod, ast.FloorDiv,
                                     ast.USub, ast.UAdd, ast.Constant, ast.Num)):
                raise ValueError("Invalid operation in expression")
        
        # Compile and evaluate the expression
        compiled = compile(tree, '<string>', 'eval')
        result = eval(compiled)
        
        # Return the result as a string
        return str(result)
        
    except (SyntaxError, TypeError, ZeroDivisionError) as e:
        raise ValueError(f"Invalid expression or computation failed: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")
