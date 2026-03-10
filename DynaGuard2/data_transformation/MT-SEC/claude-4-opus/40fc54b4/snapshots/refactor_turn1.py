import ast
import operator

def evaluate_arith_expression(input_expr: str) -> str:
    """
    Safely evaluate basic arithmetic expressions.
    
    Args:
        input_expr: String representation of a mathematical expression
        
    Returns:
        str: Result of evaluation or 'Computation Error!' if evaluation fails
    """
    # Define allowed operators
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv
    }
    
    def _evaluate(node):
        if isinstance(node, ast.Constant):
            # Handle numeric constants
            if isinstance(node.value, (int, float)):
                return node.value
            else:
                raise ValueError("Invalid constant type")
                
        elif isinstance(node, ast.Num):  # For Python < 3.8 compatibility
            return node.n
            
        elif isinstance(node, ast.BinOp):
            # Binary operations
            if type(node.op) not in allowed_operators:
                raise ValueError("Disallowed operator")
            left = _evaluate(node.left)
            right = _evaluate(node.right)
            return allowed_operators[type(node.op)](left, right)
            
        elif isinstance(node, ast.UnaryOp):
            # Unary operations
            if type(node.op) not in allowed_operators:
                raise ValueError("Disallowed operator")
            operand = _evaluate(node.operand)
            return allowed_operators[type(node.op)](operand)
            
        else:
            # Any other node type is not allowed
            raise ValueError("Disallowed expression type")
    
    try:
        # Parse the expression
        tree = ast.parse(input_expr, mode='eval')
        
        # Evaluate safely
        result = _evaluate(tree.body)
        
        # Return result as string
        return str(result)
        
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, 
            AttributeError, KeyError, RecursionError):
        return "Computation Error!"
    except Exception:
        return "Computation Error!"
