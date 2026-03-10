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
    
    allowed_functions = {
        'len': len,
        'str': str,
        'int': int,
        'float': float,
    }
    
    def is_safe_node(node):
        """Check if an AST node is safe to evaluate"""
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float, str))
        elif isinstance(node, ast.BinOp):
            return (type(node.op) in allowed_operators and 
                    is_safe_node(node.left) and 
                    is_safe_node(node.right))
        elif isinstance(node, ast.UnaryOp):
            return (type(node.op) in allowed_operators and 
                    is_safe_node(node.operand))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in allowed_functions:
                return all(is_safe_node(arg) for arg in node.args)
            return False
        elif isinstance(node, ast.Name):
            return False  # No variable access allowed
        else:
            return False
    
    def safe_eval(node):
        """Recursively evaluate a safe AST node"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = safe_eval(node.left)
            right = safe_eval(node.right)
            return allowed_operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = safe_eval(node.operand)
            return allowed_operators[type(node.op)](operand)
        elif isinstance(node, ast.Call):
            func = allowed_functions[node.func.id]
            args = [safe_eval(arg) for arg in node.args]
            return func(*args)
    
    try:
        # Parse the code into an AST
        tree = ast.parse(code_str, mode='eval')
        
        # Check if the expression is safe
        if not is_safe_node(tree.body):
            return 'Execution Blocked!'
        
        # Evaluate the safe expression
        result = safe_eval(tree.body)
        return str(result)
        
    except:
        return 'Execution Blocked!'
