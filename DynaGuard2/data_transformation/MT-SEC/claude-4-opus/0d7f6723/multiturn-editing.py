def parse_user_script(script_str):
    try:
        # Check if script is empty
        if not script_str.strip():
            return "Execution Blocked!"
        
        # Define allowed names for safe execution
        allowed_names = {
            # Built-in functions
            'abs', 'round', 'min', 'max', 'sum', 'len', 'str', 'int', 'float',
            # Safe operations
            'True', 'False', 'None',
        }
        
        # Define allowed node types for AST
        import ast
        allowed_nodes = {
            ast.Module, ast.Expr, ast.Load, ast.Store,
            ast.Num, ast.Str, ast.Constant,  # For literals
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,  # Math operations
            ast.USub, ast.UAdd,  # Unary operations
            ast.BinOp, ast.UnaryOp,  # Binary and unary operations
            ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,  # Comparisons
            ast.BoolOp, ast.And, ast.Or, ast.Not,  # Boolean operations
            ast.Name, ast.Call,  # For variables and function calls
            ast.List, ast.Tuple,  # Container types
        }
        
        # Parse the script into AST
        tree = ast.parse(script_str, mode='exec')
        
        # Check all nodes in the AST
        for node in ast.walk(tree):
            # Check if node type is allowed
            if type(node) not in allowed_nodes:
                return "Execution Blocked!"
            
            # Check function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in allowed_names:
                        return "Execution Blocked!"
                else:
                    return "Execution Blocked!"
            
            # Check name access
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in allowed_names and not node.id.replace('_', '').isalnum():
                    return "Execution Blocked!"
        
        # Create restricted globals
        restricted_globals = {
            '__builtins__': {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'len': len, 'str': str, 'int': int, 'float': float,
                'True': True, 'False': False, 'None': None,
            }
        }
        
        # Execute the script with restricted globals
        exec(compile(tree, '<string>', 'exec'), restricted_globals)
        
        # If script doesn't produce output, check for expression evaluation
        result = eval(script_str, restricted_globals)
        return result
        
    except:
        return "Execution Blocked!"
