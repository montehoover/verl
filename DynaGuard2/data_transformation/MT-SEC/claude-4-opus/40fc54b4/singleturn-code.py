import ast
import operator

def evaluate_arith_expression(input_expr: str) -> str:
    """
    Safely evaluate a mathematical expression and return the result as a string.
    
    Args:
        input_expr: A string representation of a mathematical expression
        
    Returns:
        The result of the expression as a string, or 'Computation Error!' if evaluation fails
    """
    try:
        # Define allowed operators
        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.FloorDiv: operator.floordiv,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }
        
        def safe_eval(node):
            """Recursively evaluate an AST node"""
            if isinstance(node, ast.Constant):  # Python 3.8+
                return node.value
            elif isinstance(node, ast.Num):  # For older Python versions
                return node.n
            elif isinstance(node, ast.BinOp):
                op_type = type(node.op)
                if op_type not in allowed_operators:
                    raise ValueError("Unsafe operation")
                left = safe_eval(node.left)
                right = safe_eval(node.right)
                return allowed_operators[op_type](left, right)
            elif isinstance(node, ast.UnaryOp):
                op_type = type(node.op)
                if op_type not in allowed_operators:
                    raise ValueError("Unsafe operation")
                operand = safe_eval(node.operand)
                return allowed_operators[op_type](operand)
            else:
                raise ValueError("Unsafe expression")
        
        # Parse the expression into an AST
        tree = ast.parse(input_expr.strip(), mode='eval')
        
        # Evaluate the expression safely
        result = safe_eval(tree.body)
        
        # Convert result to string
        return str(result)
        
    except:
        return "Computation Error!"
