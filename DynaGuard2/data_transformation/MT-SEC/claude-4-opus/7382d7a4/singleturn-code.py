import ast
import operator

def execute_user_code(script_code: str) -> str:
    """
    Securely evaluates a user-supplied Python script limited to basic arithmetic and string operations.
    
    Args:
        script_code: str - A user-supplied Python script limited to simple arithmetic and string operations
        
    Returns:
        str - The result of evaluating the script or 'Execution Blocked!' if unsafe
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
        ast.Num,  # For Python < 3.8 compatibility
        ast.Constant,  # For Python >= 3.8
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
        """Recursively check if a node and all its children are safe."""
        if not isinstance(node, allowed_nodes):
            return False
        
        for child in ast.iter_child_nodes(node):
            if not is_safe_node(child):
                return False
        
        return True
    
    def evaluate_node(node):
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Constant):
            # Check if constant is a basic type (int, float, str)
            if isinstance(node.value, (int, float, str)):
                return node.value
            else:
                raise ValueError("Unsupported constant type")
        
        # For Python < 3.8 compatibility
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Str):
            return node.s
        
        elif isinstance(node, ast.BinOp):
            left = evaluate_node(node.left)
            right = evaluate_node(node.right)
            op = allowed_operators.get(type(node.op))
            if op is None:
                raise ValueError("Unsupported binary operation")
            
            # Handle string concatenation
            if isinstance(left, str) and isinstance(right, str) and isinstance(node.op, ast.Add):
                return left + right
            # Handle string multiplication
            elif isinstance(left, str) and isinstance(right, int) and isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(left, int) and isinstance(right, str) and isinstance(node.op, ast.Mult):
                return left * right
            # Handle numeric operations
            elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
                return op(left, right)
            else:
                raise ValueError("Invalid operation between types")
        
        elif isinstance(node, ast.UnaryOp):
            operand = evaluate_node(node.operand)
            op = allowed_operators.get(type(node.op))
            if op is None:
                raise ValueError("Unsupported unary operation")
            if not isinstance(operand, (int, float)):
                raise ValueError("Unary operations only supported on numbers")
            return op(operand)
        
        else:
            raise ValueError("Unsupported node type")
    
    try:
        # Parse the script into an AST
        tree = ast.parse(script_code, mode='eval')
        
        # Check if all nodes are safe
        if not is_safe_node(tree):
            return 'Execution Blocked!'
        
        # Evaluate the expression
        result = evaluate_node(tree.body)
        
        # Convert result to string
        return str(result)
        
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError):
        return 'Execution Blocked!'
    except Exception:
        return 'Execution Blocked!'
