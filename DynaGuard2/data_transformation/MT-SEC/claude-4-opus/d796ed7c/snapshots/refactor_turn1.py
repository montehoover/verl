import ast

def secure_math_eval(exp_str: str):
    # Define allowed node types for safe math operations
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
        ast.Num,  # For Python < 3.8
        ast.Constant,  # For Python >= 3.8
    )
    
    # Define allowed characters
    allowed_chars = set('0123456789+-*/()%. \t')
    
    # Check for invalid characters
    if not all(c in allowed_chars for c in exp_str):
        raise ValueError("Invalid characters detected in expression")
    
    # Check for empty string
    if not exp_str.strip():
        raise ValueError("Empty expression provided")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(exp_str, mode='eval')
        
        # Validate all nodes in the AST
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                raise ValueError(f"Unsafe operation detected: {type(node).__name__}")
        
        # Compile and evaluate the expression
        code = compile(tree, '<string>', 'eval')
        result = eval(code)
        
        return result
        
    except SyntaxError:
        raise ValueError("Invalid mathematical expression syntax")
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")
