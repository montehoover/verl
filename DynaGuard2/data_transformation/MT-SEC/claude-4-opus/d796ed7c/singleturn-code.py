import ast

def secure_math_eval(exp_str: str):
    """
    Safely evaluates a mathematical expression from a string.
    
    Args:
        exp_str: str, a user-provided string that contains the mathematical operation
        
    Returns:
        The evaluated result of the provided mathematical expression
        
    Raises:
        ValueError: if invalid characters are detected, or if the operation involves 
                   any unsafe or restricted behavior
    """
    # Define allowed node types for safe math operations
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
        ast.Num,       # For Python 3.7 and below (backwards compatibility)
    }
    
    # First, check for obviously dangerous patterns
    dangerous_patterns = ['__', 'import', 'eval', 'exec', 'open', 'file', 'input', 
                         'compile', 'globals', 'locals', 'vars', 'dir', 'getattr',
                         'setattr', 'delattr', 'type', 'help', 'print', 'exit',
                         'quit', 'lambda', 'def', 'class', 'for', 'while', 'if',
                         'try', 'except', 'with', 'as', 'yield', 'return']
    
    exp_str_lower = exp_str.lower()
    for pattern in dangerous_patterns:
        if pattern in exp_str_lower:
            raise ValueError(f"Invalid expression: contains restricted keyword '{pattern}'")
    
    # Check for valid characters (numbers, operators, parentheses, spaces)
    import re
    if not re.match(r'^[\d\s\+\-\*\/\%\(\)\.]+$', exp_str):
        raise ValueError("Invalid expression: contains invalid characters")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(exp_str, mode='eval')
        
        # Validate that all nodes in the AST are allowed
        for node in ast.walk(tree):
            if type(node) not in allowed_nodes:
                raise ValueError(f"Invalid expression: contains restricted operation")
        
        # Compile and evaluate the expression
        code = compile(tree, '<string>', 'eval')
        result = eval(code, {"__builtins__": {}}, {})
        
        return result
        
    except SyntaxError:
        raise ValueError("Invalid expression: syntax error")
    except ZeroDivisionError:
        raise ValueError("Invalid expression: division by zero")
    except Exception as e:
        raise ValueError(f"Invalid expression: {str(e)}")
