import ast
import operator

def math_expression_safety_check(math_input: str) -> str:
    """
    Safely evaluates a Python mathematical expression given as a string input.
    
    Args:
        math_input: str, a string containing a Python mathematical expression
        
    Returns:
        str, the result of the safe evaluation, or 'Unsafe Code Detected!' if dangerous content is found
    """
    
    # Define allowed operators
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
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,  # For Python 3.8+
        ast.Num,       # For older Python versions
    )
    
    def safe_eval(node):
        """Recursively evaluate an AST node safely."""
        if isinstance(node, ast.Expression):
            return safe_eval(node.body)
        
        elif isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
            
        elif isinstance(node, ast.Num):  # Older Python versions
            return node.n
            
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in allowed_operators:
                raise ValueError("Unsafe operator")
            left = safe_eval(node.left)
            right = safe_eval(node.right)
            return allowed_operators[type(node.op)](left, right)
            
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in allowed_operators:
                raise ValueError("Unsafe operator")
            operand = safe_eval(node.operand)
            return allowed_operators[type(node.op)](operand)
            
        else:
            raise ValueError("Unsafe node type")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(math_input, mode='eval')
        
        # Check if all nodes are allowed
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes) and type(node) not in allowed_operators.keys():
                return "Unsafe Code Detected!"
        
        # Evaluate the expression safely
        result = safe_eval(tree)
        
        # Convert result to string
        return str(result)
        
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError):
        return "Unsafe Code Detected!"
    except Exception:
        return "Unsafe Code Detected!"
