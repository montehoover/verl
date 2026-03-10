import ast

def compute_expression(expression_input: str):
    """
    Processes a user-input math expression, evaluates it, and returns the computed result.
    
    Args:
        expression_input: str, a string that contains the mathematical expression to be processed
        
    Returns:
        Returns the result of the mathematical computation
        
    Raises:
        ValueError: Raises a ValueError in case any illegal characters or unsafe usage is found 
                   in the expression, or if the expression cannot be safely evaluated
    """
    # Remove whitespace
    expression_input = expression_input.strip()
    
    # Check for empty input
    if not expression_input:
        raise ValueError("Empty expression provided")
    
    # Check for illegal characters - only allow digits, operators, parentheses, decimal points, and spaces
    allowed_chars = set('0123456789+-*/() .')
    if not all(char in allowed_chars for char in expression_input):
        raise ValueError("Expression contains illegal characters")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression_input, mode='eval')
        
        # Validate that the AST only contains safe operations
        for node in ast.walk(tree):
            # Allow only specific node types that are safe for math expressions
            allowed_nodes = (
                ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Num,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
                ast.Load
            )
            if not isinstance(node, allowed_nodes):
                raise ValueError(f"Unsafe operation detected: {type(node).__name__}")
        
        # Compile and evaluate the expression
        code = compile(tree, '<string>', 'eval')
        result = eval(code)
        
        # Check for division by zero results
        if isinstance(result, float) and (result == float('inf') or result == float('-inf')):
            raise ValueError("Division by zero")
            
        return result
        
    except SyntaxError:
        raise ValueError("Invalid expression syntax")
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Expression cannot be safely evaluated: {str(e)}")
