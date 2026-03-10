import ast
import operator

def parse_user_script(script_str: str) -> str:
    """
    Safely evaluate a user-provided Python script containing only simple arithmetic and string operations.
    
    Args:
        script_str: A string containing a tiny Python script with only basic arithmetic or string operations
        
    Returns:
        The result of the script execution as a string, or 'Execution Blocked!' if unsafe
    """
    
    # Define allowed operators
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
    
    def safe_eval(node):
        """Recursively evaluate an AST node safely."""
        if isinstance(node, ast.Constant):
            # Allow numbers and strings
            if isinstance(node.value, (int, float, str)):
                return node.value
            else:
                raise ValueError("Unsafe constant type")
                
        elif isinstance(node, ast.Num):  # For Python < 3.8 compatibility
            return node.n
            
        elif isinstance(node, ast.Str):  # For Python < 3.8 compatibility
            return node.s
            
        elif isinstance(node, ast.BinOp):
            # Binary operations
            if type(node.op) not in allowed_ops:
                raise ValueError("Unsafe operation")
            left = safe_eval(node.left)
            right = safe_eval(node.right)
            return allowed_ops[type(node.op)](left, right)
            
        elif isinstance(node, ast.UnaryOp):
            # Unary operations
            if type(node.op) not in allowed_ops:
                raise ValueError("Unsafe operation")
            operand = safe_eval(node.operand)
            return allowed_ops[type(node.op)](operand)
            
        else:
            raise ValueError("Unsafe node type")
    
    try:
        # Parse the script
        tree = ast.parse(script_str, mode='eval')
        
        # Evaluate safely
        result = safe_eval(tree.body)
        
        # Convert result to string
        return str(result)
        
    except:
        # Any error means the script is unsafe or invalid
        return "Execution Blocked!"
