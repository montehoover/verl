import ast

def secure_eval_expression(exp: str):
    # Define allowed node types for safe mathematical expressions
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
        ast.Constant,
        ast.Num,  # For Python < 3.8 compatibility
    )
    
    # Define allowed characters (digits, operators, parentheses, decimal point, whitespace)
    allowed_chars = set('0123456789+-*/()%. \t\n')
    
    # Check for invalid characters
    if not all(c in allowed_chars for c in exp):
        raise ValueError("Invalid characters detected in expression")
    
    # Check for empty expression
    if not exp.strip():
        raise ValueError("Empty expression")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(exp, mode='eval')
        
        # Validate that all nodes in the AST are allowed
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                raise ValueError(f"Unsafe operation detected: {type(node).__name__}")
        
        # Compile and evaluate the expression
        code = compile(tree, '<string>', 'eval')
        result = eval(code, {"__builtins__": {}})
        
        return result
        
    except SyntaxError:
        raise ValueError("Invalid mathematical expression syntax")
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")
