import ast

def perform_safe_math(expression: str):
    """
    Evaluate a user-provided string that contains a mathematical expression and return the result.
    
    Args:
        expression: str, a string containing the mathematical expression to be evaluated
        
    Returns:
        Returns the result of evaluating the given mathematical expression
        
    Raises:
        ValueError: if any invalid input, such as unsafe characters or operations, is detected
                   or if the expression is improperly formatted or handled
    """
    # Check if expression is empty or None
    if not expression or not expression.strip():
        raise ValueError("Expression cannot be empty")
    
    # Define allowed node types for safe evaluation
    allowed_nodes = (
        ast.Expression,
        ast.Constant,
        ast.Num,  # For Python < 3.8 compatibility
        ast.Str,  # For Python < 3.8 compatibility
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    )
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')
        
        # Validate that the AST only contains allowed operations
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                raise ValueError(f"Unsafe operation detected: {type(node).__name__}")
        
        # Compile and evaluate the expression
        compiled = compile(tree, '<string>', 'eval')
        result = eval(compiled, {"__builtins__": {}})
        
        return result
        
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {str(e)}")
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")
