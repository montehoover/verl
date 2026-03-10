import ast

def safe_execute_operation(operation: str):
    """
    Safely execute a mathematical operation provided as a string.
    
    Args:
        operation: str - The mathematical operation to be evaluated
        
    Returns:
        The result of the evaluated operation
        
    Raises:
        ValueError: If unsafe or invalid characters are detected or if the evaluation logic encounters any issues
    """
    # Define allowed node types for safe evaluation
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.FloorDiv,
        ast.USub,
        ast.UAdd,
        ast.Constant,  # For Python 3.8+
        ast.Num,       # For older Python versions
    )
    
    try:
        # Parse the operation string into an AST
        tree = ast.parse(operation, mode='eval')
        
        # Validate that all nodes in the AST are allowed
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                raise ValueError(f"Unsafe operation detected: {type(node).__name__} is not allowed")
        
        # If validation passes, compile and evaluate the expression
        code = compile(tree, '<string>', 'eval')
        result = eval(code)
        
        return result
        
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax in operation: {e}")
    except Exception as e:
        raise ValueError(f"Error evaluating operation: {e}")
