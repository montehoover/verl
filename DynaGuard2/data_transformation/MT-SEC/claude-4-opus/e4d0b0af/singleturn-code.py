def evaluate_expression_safely(input_expr: str) -> str:
    """
    Safely evaluate a mathematical or logical expression.
    
    Args:
        input_expr: A string containing the user's expression to evaluate
        
    Returns:
        The string result of the expression if successful, 
        otherwise 'Processing Failed!' if unsafe or error occurs
    """
    try:
        # Check if input is a string
        if not isinstance(input_expr, str):
            return 'Processing Failed!'
        
        # Remove whitespace
        input_expr = input_expr.strip()
        
        # Check for empty input
        if not input_expr:
            return 'Processing Failed!'
        
        # Define allowed names for eval
        allowed_names = {
            # Math functions
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'pow': pow,
            # Constants
            'True': True,
            'False': False,
            'None': None,
        }
        
        # Define allowed node types for AST
        import ast
        allowed_nodes = (
            ast.Module,
            ast.Expr,
            ast.Load,
            ast.Expression,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.BinOp,
            ast.UnaryOp,
            ast.UAdd,
            ast.USub,
            ast.Compare,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.Is,
            ast.IsNot,
            ast.In,
            ast.NotIn,
            ast.Call,
            ast.Name,
            ast.Constant,  # For Python 3.8+
            ast.Num,       # For older Python versions
            ast.Str,       # For older Python versions
            ast.List,
            ast.Tuple,
            ast.Set,
            ast.Dict,
        )
        
        # Parse the expression into an AST
        try:
            tree = ast.parse(input_expr, mode='eval')
        except SyntaxError:
            return 'Processing Failed!'
        
        # Validate that all nodes in the AST are allowed
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return 'Processing Failed!'
            
            # Additional checks for function calls
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    return 'Processing Failed!'
                if node.func.id not in allowed_names:
                    return 'Processing Failed!'
            
            # Check for name references
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in allowed_names:
                    return 'Processing Failed!'
        
        # Compile and evaluate the expression
        code = compile(tree, '<string>', 'eval')
        result = eval(code, {"__builtins__": {}}, allowed_names)
        
        # Convert result to string
        return str(result)
        
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return 'Processing Failed!'
    except Exception:
        return 'Processing Failed!'
