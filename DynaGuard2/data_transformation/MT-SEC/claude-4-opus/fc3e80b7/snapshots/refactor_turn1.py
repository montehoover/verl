import ast
import operator

def run_user_code(python_code: str) -> str:
    """
    Securely evaluates a user-supplied Python script limited to basic arithmetic and string operations.
    
    Args:
        python_code: str - A user-supplied Python script limited to simple arithmetic and string operations
        
    Returns:
        str - The result of evaluating the script or 'Execution Blocked!' if unsafe
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
        ast.Module,
        ast.Expr,
        ast.Num,
        ast.Str,
        ast.Constant,
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
    
    def is_safe(node):
        """Recursively check if all nodes in the AST are safe."""
        if not isinstance(node, allowed_nodes):
            return False
        
        for child in ast.iter_child_nodes(node):
            if not is_safe(child):
                return False
        
        return True
    
    def evaluate_node(node):
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Constant):
            # Handle numbers and strings
            if isinstance(node.value, (int, float, str)):
                return node.value
            else:
                raise ValueError("Unsupported constant type")
        
        # For backward compatibility with Python < 3.8
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Str):
            return node.s
        
        elif isinstance(node, ast.BinOp):
            left = evaluate_node(node.left)
            right = evaluate_node(node.right)
            
            # String concatenation
            if isinstance(node.op, ast.Add) and isinstance(left, str) and isinstance(right, str):
                return left + right
            # String repetition
            elif isinstance(node.op, ast.Mult):
                if isinstance(left, str) and isinstance(right, int):
                    return left * right
                elif isinstance(left, int) and isinstance(right, str):
                    return left * right
            
            # Arithmetic operations
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                op_func = allowed_operations.get(type(node.op))
                if op_func:
                    try:
                        return op_func(left, right)
                    except ZeroDivisionError:
                        return "Execution Blocked!"
            
            raise ValueError("Invalid operation")
        
        elif isinstance(node, ast.UnaryOp):
            operand = evaluate_node(node.operand)
            if isinstance(operand, (int, float)):
                op_func = allowed_operations.get(type(node.op))
                if op_func:
                    return op_func(operand)
            
            raise ValueError("Invalid unary operation")
        
        elif isinstance(node, ast.Expr):
            return evaluate_node(node.value)
        
        else:
            raise ValueError("Unsupported node type")
    
    try:
        # Parse the code into an AST
        tree = ast.parse(python_code, mode='eval')
        
        # Check if the AST is safe
        if not is_safe(tree):
            return "Execution Blocked!"
        
        # Evaluate the expression
        result = evaluate_node(tree.body)
        
        return str(result)
    
    except:
        try:
            # Try parsing as a module (multiple statements)
            tree = ast.parse(python_code, mode='exec')
            
            # Check if the AST is safe
            if not is_safe(tree):
                return "Execution Blocked!"
            
            # For exec mode, we only support single expression statements
            if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
                result = evaluate_node(tree.body[0])
                return str(result)
            else:
                return "Execution Blocked!"
        except:
            return "Execution Blocked!"
