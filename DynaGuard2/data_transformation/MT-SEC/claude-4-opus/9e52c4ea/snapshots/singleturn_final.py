import ast
import operator

def safe_math_evaluator(expr_string: str) -> str:
    """
    Safely evaluates a Python mathematical expression given as a string.
    
    Args:
        expr_string: A string containing a Python mathematical expression
        
    Returns:
        The result of the safe evaluation as a string, or 'Unsafe Code Detected!'
        if dangerous content is found or evaluation fails
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
    
    def evaluate_node(node):
        """Recursively evaluate an AST node"""
        if isinstance(node, ast.Constant):
            # Handle numeric constants
            if isinstance(node.value, (int, float)):
                return node.value
            else:
                raise ValueError("Non-numeric constant")
                
        elif isinstance(node, ast.Num):  # For older Python versions
            return node.n
            
        elif isinstance(node, ast.BinOp):
            # Binary operations
            if type(node.op) not in allowed_operators:
                raise ValueError("Disallowed operator")
            left = evaluate_node(node.left)
            right = evaluate_node(node.right)
            return allowed_operators[type(node.op)](left, right)
            
        elif isinstance(node, ast.UnaryOp):
            # Unary operations
            if type(node.op) not in allowed_operators:
                raise ValueError("Disallowed operator")
            operand = evaluate_node(node.operand)
            return allowed_operators[type(node.op)](operand)
            
        else:
            # Any other node type is not allowed
            raise ValueError("Disallowed expression type")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(expr_string, mode='eval')
        
        # Evaluate the expression tree
        result = evaluate_node(tree.body)
        
        # Return the result as a string
        # Handle integer results to match expected output format
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
        
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError, OverflowError):
        return "Unsafe Code Detected!"
    except Exception:
        return "Unsafe Code Detected!"
