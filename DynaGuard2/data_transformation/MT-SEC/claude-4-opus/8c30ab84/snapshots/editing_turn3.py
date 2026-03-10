import ast

def compute_expression(expression_input):
    # Check for potentially unsafe patterns
    unsafe_patterns = ['import', 'exec', 'eval', '__', 'open', 'file', 'input', 'compile', 'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr', 'delattr', 'type', 'help', 'reload', 'id', 'repr']
    
    expression_lower = expression_input.lower()
    for pattern in unsafe_patterns:
        if pattern in expression_lower:
            raise ValueError(f"Unsafe expression: contains '{pattern}'")
    
    # Check for other potentially dangerous characters
    if any(char in expression_input for char in [';', '\\', '`', '$', '&', '|', '!', '~', '[', ']', '{', '}', '"', "'"]):
        raise ValueError("Expression contains potentially unsafe characters")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression_input, mode='eval')
        
        # Check that the AST only contains safe operations
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, 
                               ast.ClassDef, ast.Return, ast.Delete, ast.Assign, ast.AugAssign,
                               ast.For, ast.AsyncFor, ast.While, ast.If, ast.With, ast.AsyncWith,
                               ast.Raise, ast.Try, ast.Assert, ast.Global, ast.Nonlocal, ast.Pass,
                               ast.Break, ast.Continue, ast.Lambda, ast.Yield, ast.YieldFrom,
                               ast.Call, ast.Attribute)):
                raise ValueError("Expression contains unsafe operations")
        
        # Compile and evaluate the expression
        compiled = compile(tree, '<string>', 'eval')
        result = eval(compiled, {"__builtins__": {}}, {})
        
        # Ensure the result is numeric
        if not isinstance(result, (int, float, complex)):
            raise ValueError("Expression must evaluate to a number")
        
        return result
        
    except (SyntaxError, TypeError, ZeroDivisionError) as e:
        raise ValueError(f"Invalid expression: {str(e)}")
