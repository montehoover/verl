import ast

def secure_eval_expression(exp: str):
    """
    Safely evaluates a mathematical expression from user input.
    
    Args:
        exp: str - A user-provided string that contains the mathematical operation
        
    Returns:
        The evaluated result of the provided mathematical expression
        
    Raises:
        ValueError: If invalid characters are detected, or if the operation 
                   involves any unsafe or restricted behavior
    """
    # Remove whitespace for consistent processing
    exp = exp.strip()
    
    # Check if the expression is empty
    if not exp:
        raise ValueError("Empty expression provided")
    
    # Define allowed characters for mathematical expressions
    allowed_chars = set('0123456789+-*/()._ \t\n')
    
    # Check for invalid characters
    if not all(c in allowed_chars for c in exp):
        raise ValueError(f"Invalid characters detected in expression: {exp}")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(exp, mode='eval')
        
        # Define allowed node types for safe evaluation
        allowed_nodes = {
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Constant,  # For Python 3.8+
            ast.Num,       # For backward compatibility
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Mod,
            ast.FloorDiv,
        }
        
        # Validate that all nodes in the AST are allowed
        for node in ast.walk(tree):
            if type(node) not in allowed_nodes:
                # Special handling for Name nodes (variables) - not allowed
                if isinstance(node, ast.Name):
                    raise ValueError(f"Variable names are not allowed: {node.id}")
                # Special handling for Call nodes (function calls) - not allowed
                elif isinstance(node, ast.Call):
                    raise ValueError("Function calls are not allowed")
                else:
                    raise ValueError(f"Unsafe operation detected: {type(node).__name__}")
        
        # Compile and evaluate the expression
        compiled = compile(tree, '<string>', 'eval')
        result = eval(compiled, {"__builtins__": {}}, {})
        
        return result
        
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax in expression: {e}")
    except ZeroDivisionError:
        raise ValueError("Division by zero error")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {e}")
