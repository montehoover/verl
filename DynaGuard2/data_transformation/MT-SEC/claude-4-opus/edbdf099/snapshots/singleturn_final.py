import ast
import operator

def process_user_code(code_script: str) -> str:
    """
    Securely parse and execute a Python script containing basic arithmetic or string operations.
    
    Args:
        code_script: A Python script submitted by the user containing simple arithmetic or string manipulations
        
    Returns:
        The evaluated result of the script, or 'Execution Blocked!' if the script contains dangerous actions
    """
    
    # Define allowed operations
    allowed_operations = {
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
        ast.Expression,
        ast.Constant,
        ast.Num,  # For Python < 3.8 compatibility
        ast.Str,  # For Python < 3.8 compatibility
        ast.BinOp,
        ast.UnaryOp,
    )
    
    def is_safe_node(node):
        """Check if a node is safe to evaluate."""
        if isinstance(node, allowed_nodes):
            return True
        return False
    
    def evaluate_node(node):
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Constant):
            # Only allow basic types
            if isinstance(node.value, (int, float, str)):
                return node.value
            else:
                raise ValueError("Unsafe constant type")
                
        # For Python < 3.8 compatibility
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Str):
            return node.s
            
        elif isinstance(node, ast.BinOp):
            left = evaluate_node(node.left)
            right = evaluate_node(node.right)
            op = type(node.op)
            
            if op in allowed_operations:
                try:
                    return allowed_operations[op](left, right)
                except Exception:
                    raise ValueError("Invalid operation")
            else:
                raise ValueError("Unsafe operation")
                
        elif isinstance(node, ast.UnaryOp):
            operand = evaluate_node(node.operand)
            op = type(node.op)
            
            if op in allowed_operations:
                return allowed_operations[op](operand)
            else:
                raise ValueError("Unsafe operation")
                
        else:
            raise ValueError("Unsafe node type")
    
    try:
        # Parse the code
        tree = ast.parse(code_script, mode='eval')
        
        # Check all nodes in the tree are safe
        for node in ast.walk(tree):
            if not is_safe_node(node):
                return "Execution Blocked!"
        
        # Evaluate the expression
        result = evaluate_node(tree.body)
        
        # Convert result to string
        return str(result)
        
    except Exception:
        return "Execution Blocked!"
