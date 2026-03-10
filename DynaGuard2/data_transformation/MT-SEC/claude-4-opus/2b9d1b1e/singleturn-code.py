import ast
import operator

def evaluate_user_code(code_str: str) -> str:
    """
    Safely evaluate a user-supplied Python script limited to basic arithmetic and string operations.
    
    Args:
        code_str: A user-supplied Python script limited to simple arithmetic and string operations
        
    Returns:
        The result of evaluating the script or 'Execution Blocked!' if unsafe
    """
    # Define allowed operations
    allowed_operators = {
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
        ast.Load,
        ast.Expression,
        ast.Constant,  # For Python 3.8+
        ast.Num,       # For older Python versions
        ast.Str,       # For older Python versions
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
        """Check if a node is safe to evaluate"""
        if not isinstance(node, allowed_nodes):
            return False
        
        # Recursively check child nodes
        for child in ast.iter_child_nodes(node):
            if not is_safe_node(child):
                return False
                
        return True
    
    def eval_node(node):
        """Evaluate a safe AST node"""
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Older Python versions
            return node.n
        elif isinstance(node, ast.Str):  # Older Python versions
            return node.s
        elif isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            op_func = allowed_operators.get(type(node.op))
            if op_func:
                return op_func(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            op_func = allowed_operators.get(type(node.op))
            if op_func:
                return op_func(operand)
        elif isinstance(node, ast.Expression):
            return eval_node(node.body)
        elif isinstance(node, ast.Expr):
            return eval_node(node.value)
        
        raise ValueError("Unsupported operation")
    
    try:
        # Parse the code into an AST
        tree = ast.parse(code_str, mode='eval')
        
        # Check if the AST is safe
        if not is_safe_node(tree):
            return 'Execution Blocked!'
        
        # Evaluate the safe AST
        result = eval_node(tree)
        
        # Convert result to string
        return str(result)
        
    except:
        # Any error in parsing or evaluation results in blocking
        return 'Execution Blocked!'
