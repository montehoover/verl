import ast
import operator

def process_user_expression(expr: str) -> str:
    """
    Safely evaluate user-provided mathematical and logical expressions.
    
    Args:
        expr: A string containing the user's expression that needs evaluation
        
    Returns:
        The result of the evaluated expression as a string, or 'Processing Failed!'
        if the input is invalid or a security risk is detected
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
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.And: lambda x, y: x and y,
        ast.Or: lambda x, y: x or y,
        ast.Not: operator.not_,
    }
    
    def safe_eval(node):
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Constant):
            # Handle literal values (numbers, strings, booleans, None)
            return node.value
        elif isinstance(node, ast.Num):  # For Python < 3.8 compatibility
            return node.n
        elif isinstance(node, ast.Str):  # For Python < 3.8 compatibility
            return node.s
        elif isinstance(node, ast.BinOp):
            # Binary operations (e.g., addition, subtraction)
            op_type = type(node.op)
            if op_type not in allowed_operators:
                raise ValueError(f"Operator {op_type} not allowed")
            left = safe_eval(node.left)
            right = safe_eval(node.right)
            return allowed_operators[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            # Unary operations (e.g., negation)
            op_type = type(node.op)
            if op_type not in allowed_operators:
                raise ValueError(f"Operator {op_type} not allowed")
            operand = safe_eval(node.operand)
            return allowed_operators[op_type](operand)
        elif isinstance(node, ast.Compare):
            # Comparison operations
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise ValueError("Complex comparisons not allowed")
            op_type = type(node.ops[0])
            if op_type not in allowed_operators:
                raise ValueError(f"Operator {op_type} not allowed")
            left = safe_eval(node.left)
            right = safe_eval(node.comparators[0])
            return allowed_operators[op_type](left, right)
        elif isinstance(node, ast.BoolOp):
            # Boolean operations (and, or)
            op_type = type(node.op)
            if op_type not in allowed_operators:
                raise ValueError(f"Operator {op_type} not allowed")
            values = [safe_eval(value) for value in node.values]
            result = values[0]
            for value in values[1:]:
                result = allowed_operators[op_type](result, value)
            return result
        elif isinstance(node, ast.NameConstant):  # For Python < 3.8 compatibility
            # Handle True, False, None
            return node.value
        else:
            # Reject any other node types (function calls, attributes, etc.)
            raise ValueError(f"Expression type {type(node)} not allowed")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(expr, mode='eval')
        
        # Evaluate the expression safely
        result = safe_eval(tree.body)
        
        # Return the result as a string
        return str(result)
        
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, 
            OverflowError, AttributeError, KeyError, IndexError) as e:
        # Return error message for any expected exceptions
        return "Processing Failed!"
    except Exception:
        # Catch any other unexpected exceptions
        return "Processing Failed!"
