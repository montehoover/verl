import ast
import operator

def execute_user_code(user_code: str) -> str:
    """
    Securely parse and execute a Python script containing basic arithmetic or string operations.
    
    Args:
        user_code: str - A Python script submitted by the user containing simple arithmetic or string manipulations
        
    Returns:
        str - The evaluated result of the script, or 'Execution Blocked!' if the script contains dangerous actions
    """
    # Define allowed operations
    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    # Define allowed node types
    allowed_nodes = (
        ast.Module,
        ast.Expr,
        ast.Constant,
        ast.Num,  # For Python < 3.8 compatibility
        ast.Str,  # For Python < 3.8 compatibility
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
    )
    
    def is_safe_node(node):
        """Check if a node is safe to evaluate."""
        if not isinstance(node, allowed_nodes):
            return False
        
        # Recursively check child nodes
        for child in ast.iter_child_nodes(node):
            if not is_safe_node(child):
                return False
        
        return True
    
    def eval_node(node):
        """Evaluate a safe AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, (ast.Num, ast.Str)):  # For Python < 3.8 compatibility
            return node.n if isinstance(node, ast.Num) else node.s
        elif isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            op = allowed_ops.get(type(node.op))
            if op:
                return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            op = allowed_ops.get(type(node.op))
            if op:
                return op(operand)
        elif isinstance(node, ast.Expr):
            return eval_node(node.value)
        
        raise ValueError("Unsafe operation")
    
    try:
        # Parse the user code
        tree = ast.parse(user_code, mode='exec')
        
        # Check if the AST is safe
        if not is_safe_node(tree):
            return 'Execution Blocked!'
        
        # Evaluate the safe expression
        result = None
        for node in tree.body:
            if isinstance(node, ast.Expr):
                result = eval_node(node)
        
        # Return the result as a string
        return str(result) if result is not None else ''
        
    except:
        return 'Execution Blocked!'
