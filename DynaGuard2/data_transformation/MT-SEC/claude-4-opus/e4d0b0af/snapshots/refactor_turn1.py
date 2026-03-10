def evaluate_expression_safely(input_expr: str) -> str:
    # Define allowed names for safe evaluation
    allowed_names = {
        # Math functions
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'pow': pow,
        'len': len,
        # Constants
        'True': True,
        'False': False,
        'None': None,
    }
    
    # Define allowed node types for AST
    import ast
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.BoolOp,
        ast.IfExp,
        # Operators
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        # Comparisons
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
        # Boolean operators
        ast.And,
        ast.Or,
        ast.Not,
        # Literals and names
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.List,
        ast.Tuple,
        ast.Call,
    )
    
    def is_safe_node(node):
        """Recursively check if AST node is safe"""
        if not isinstance(node, allowed_nodes):
            return False
        
        # Check function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in allowed_names:
                    return False
            else:
                return False
        
        # Check all child nodes
        for child in ast.iter_child_nodes(node):
            if not is_safe_node(child):
                return False
        
        return True
    
    try:
        # Parse the expression
        tree = ast.parse(input_expr, mode='eval')
        
        # Check if the AST is safe
        if not is_safe_node(tree):
            return 'Processing Failed!'
        
        # Compile and evaluate
        code = compile(tree, '<string>', 'eval')
        result = eval(code, {"__builtins__": {}}, allowed_names)
        
        return str(result)
        
    except:
        return 'Processing Failed!'
