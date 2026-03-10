def calculate_user_input(input_expr: str) -> str:
    """
    Safely evaluate arithmetic and logical expressions from user input.
    
    Args:
        input_expr: str - A string representing the user-supplied expression
        
    Returns:
        str - The outcome of the evaluated expression or 'Processing Failed!'
    """
    try:
        # Define allowed names for safe evaluation
        allowed_names = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'True': True,
            'False': False,
            'None': None
        }
        
        # Check for dangerous keywords and constructs
        dangerous_keywords = [
            '__', 'import', 'exec', 'eval', 'open', 'file', 'input',
            'compile', 'globals', 'locals', 'vars', 'dir', 'getattr',
            'setattr', 'delattr', 'type', 'help', 'breakpoint', 'exit',
            'quit', 'credits', 'license', 'copyright', 'print'
        ]
        
        # Convert to lowercase for case-insensitive checking
        expr_lower = input_expr.lower()
        
        # Check for dangerous keywords
        for keyword in dangerous_keywords:
            if keyword in expr_lower:
                return 'Processing Failed!'
        
        # Check for function calls (except allowed ones)
        if '(' in input_expr:
            # Extract potential function names
            import re
            func_pattern = r'(\w+)\s*\('
            matches = re.findall(func_pattern, input_expr)
            for match in matches:
                if match not in allowed_names:
                    return 'Processing Failed!'
        
        # Use compile to check syntax and restrict to expressions only
        try:
            compiled = compile(input_expr, '<user_input>', 'eval')
        except SyntaxError:
            return 'Processing Failed!'
        
        # Check for restricted node types in the AST
        import ast
        try:
            tree = ast.parse(input_expr, mode='eval')
        except:
            return 'Processing Failed!'
        
        # Define allowed node types for safe evaluation
        allowed_nodes = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare,
            ast.Constant, ast.Name, ast.Load, ast.Store,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
            ast.Mod, ast.Pow, ast.LShift, ast.RShift,
            ast.BitOr, ast.BitXor, ast.BitAnd, ast.And, ast.Or,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
            ast.Is, ast.IsNot, ast.In, ast.NotIn, ast.Not,
            ast.Invert, ast.UAdd, ast.USub, ast.Call, ast.keyword,
            ast.IfExp, ast.BoolOp
        )
        
        # For Python 3.7 compatibility, also check for Num, Str, NameConstant
        if hasattr(ast, 'Num'):
            allowed_nodes = allowed_nodes + (ast.Num, ast.Str, ast.NameConstant)
        
        # Walk through AST and check all nodes
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return 'Processing Failed!'
            
            # Additional check for Name nodes
            if isinstance(node, ast.Name):
                if node.id not in allowed_names and not node.id.replace('_', '').isalnum():
                    return 'Processing Failed!'
            
            # Check Call nodes only call allowed functions
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in allowed_names:
                        return 'Processing Failed!'
                else:
                    return 'Processing Failed!'
        
        # Evaluate the expression with restricted namespace
        result = eval(compiled, {"__builtins__": {}}, allowed_names)
        
        # Convert result to string
        return str(result)
        
    except Exception:
        return 'Processing Failed!'
