import ast

def safe_execute_operation(operation: str):
    """
    Safely execute a mathematical operation from user input.
    
    Args:
        operation: str, the mathematical operation to be evaluated
        
    Returns:
        The result of the evaluated operation
        
    Raises:
        ValueError: if unsafe or invalid characters are detected or if the evaluation logic encounters any issues
    """
    # Check for empty input
    if not operation or not operation.strip():
        raise ValueError("Operation cannot be empty")
    
    # Define allowed characters for mathematical operations
    allowed_chars = set('0123456789+-*/()., ')
    
    # Check if all characters in the operation are allowed
    if not all(char in allowed_chars for char in operation):
        raise ValueError("Invalid characters detected in operation")
    
    # Remove spaces for parsing
    operation = operation.replace(' ', '')
    
    try:
        # Parse the operation as an AST to ensure it's a valid expression
        tree = ast.parse(operation, mode='eval')
        
        # Check that the AST only contains allowed node types
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, 
                                   ast.Num, ast.Constant, ast.Add, ast.Sub, 
                                   ast.Mult, ast.Div, ast.USub, ast.UAdd)):
                raise ValueError("Invalid operation structure")
        
        # Compile and evaluate the expression
        compiled = compile(tree, '<string>', 'eval')
        result = eval(compiled)
        
        return result
        
    except (SyntaxError, TypeError, ZeroDivisionError) as e:
        raise ValueError(f"Error evaluating operation: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error: {str(e)}")
