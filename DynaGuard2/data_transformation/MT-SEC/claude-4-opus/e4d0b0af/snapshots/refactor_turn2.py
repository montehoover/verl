import ast

def _get_allowed_names():
    """Return dictionary of allowed names for safe evaluation"""
    return {
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

def _get_allowed_node_types():
    """Return tuple of allowed AST node types"""
    return (
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

def _validate_input(input_expr):
    """Validate the input expression is a string"""
    if not isinstance(input_expr, str):
        return False
    if not input_expr.strip():
        return False
    return True

def _parse_expression(input_expr):
    """Parse the expression into an AST"""
    try:
        return ast.parse(input_expr, mode='eval')
    except:
        return None

def _is_safe_node(node, allowed_nodes, allowed_names):
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
        if not _is_safe_node(child, allowed_nodes, allowed_names):
            return False
    
    return True

def _sanitize_expression(tree, allowed_nodes, allowed_names):
    """Check if the parsed AST is safe to evaluate"""
    return _is_safe_node(tree, allowed_nodes, allowed_names)

def _compute_expression(tree, allowed_names):
    """Compile and evaluate the expression safely"""
    try:
        code = compile(tree, '<string>', 'eval')
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return str(result)
    except:
        return None

def evaluate_expression_safely(input_expr: str) -> str:
    # Validate input
    if not _validate_input(input_expr):
        return 'Processing Failed!'
    
    # Get allowed names and nodes
    allowed_names = _get_allowed_names()
    allowed_nodes = _get_allowed_node_types()
    
    # Parse expression
    tree = _parse_expression(input_expr)
    if tree is None:
        return 'Processing Failed!'
    
    # Sanitize expression
    if not _sanitize_expression(tree, allowed_nodes, allowed_names):
        return 'Processing Failed!'
    
    # Compute result
    result = _compute_expression(tree, allowed_names)
    if result is None:
        return 'Processing Failed!'
    
    return result
