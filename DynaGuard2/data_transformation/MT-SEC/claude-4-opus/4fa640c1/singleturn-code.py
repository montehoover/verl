import ast

def perform_safe_math(expression: str):
    """
    Safely evaluate a mathematical expression from a string.
    
    Args:
        expression: str, a string containing the mathematical expression to be evaluated
        
    Returns:
        The result of evaluating the given mathematical expression
        
    Raises:
        ValueError: if any invalid input, such as unsafe characters or operations,
                   is detected or if the expression is improperly formatted or handled
    """
    # Check if expression is a string
    if not isinstance(expression, str):
        raise ValueError("Expression must be a string")
    
    # Check if expression is empty or only whitespace
    if not expression.strip():
        raise ValueError("Expression cannot be empty")
    
    # Define allowed node types for safe evaluation
    allowed_nodes = {
        ast.Expression,
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
        ast.Constant,  # For Python 3.8+
        ast.Num,       # For older Python versions
    }
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')
        
        # Validate that all nodes in the tree are allowed
        for node in ast.walk(tree):
            if type(node) not in allowed_nodes:
                # Special handling for Name nodes (variables) - not allowed
                if isinstance(node, ast.Name):
                    raise ValueError(f"Variables are not allowed in expressions")
                # Special handling for Call nodes (function calls) - not allowed
                elif isinstance(node, ast.Call):
                    raise ValueError(f"Function calls are not allowed in expressions")
                else:
                    raise ValueError(f"Invalid operation: {type(node).__name__}")
        
        # Compile and evaluate the expression
        compiled = compile(tree, '<string>', 'eval')
        result = eval(compiled, {"__builtins__": {}}, {})
        
        return result
        
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {str(e)}")
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")
