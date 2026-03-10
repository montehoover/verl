import ast

def evaluate_quiz_expression(quiz_expr: str) -> str:
    """
    Safely evaluate a Python expression and return the result as a string.
    
    Args:
        quiz_expr: A string representing a Python expression to evaluate
        
    Returns:
        The result of the evaluation as a string, or 'Error!' if unsafe/invalid
    """
    try:
        # Parse the expression into an AST
        tree = ast.parse(quiz_expr, mode='eval')
        
        # Check if the expression contains only safe nodes
        for node in ast.walk(tree):
            # Allow only safe node types
            if not isinstance(node, (
                ast.Expression, ast.Constant, ast.Num, ast.Str,
                ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
                ast.Mod, ast.Pow, ast.USub, ast.UAdd,
                ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
                ast.And, ast.Or, ast.Not, ast.Is, ast.IsNot,
                ast.In, ast.NotIn, ast.List, ast.Tuple, ast.Dict,
                ast.Name, ast.Load
            )):
                return 'Error!'
        
        # If we have Name nodes, only allow safe built-in names
        safe_names = {'True', 'False', 'None'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in safe_names:
                return 'Error!'
        
        # Compile and evaluate the expression
        code = compile(tree, '<string>', 'eval')
        result = eval(code, {'__builtins__': {}}, {})
        
        # Convert result to string and return
        return str(result)
        
    except Exception:
        # Any error during parsing, compilation, or evaluation
        return 'Error!'
