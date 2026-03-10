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
    ALLOWED_OPERATORS = {
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
    
    ALLOWED_COMPARISONS = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }
    
    def safe_eval(node):
        """Recursively evaluate an AST node safely."""
        
        if isinstance(node, ast.Constant):
            # Allow basic literals (numbers, strings)
            if isinstance(node.value, (int, float, str)):
                return node.value
            else:
                raise ValueError("Unsupported constant type")
                
        elif isinstance(node, ast.BinOp):
            # Binary operations like addition, multiplication
            if type(node.op) in ALLOWED_OPERATORS:
                left = safe_eval(node.left)
                right = safe_eval(node.right)
                op_func = ALLOWED_OPERATORS[type(node.op)]
                
                # Special handling for string multiplication
                if isinstance(left, str) and isinstance(right, int):
                    return left * right
                elif isinstance(left, int) and isinstance(right, str):
                    return left * right
                    
                return op_func(left, right)
            else:
                raise ValueError("Unsupported operation")
                
        elif isinstance(node, ast.UnaryOp):
            # Unary operations like negation
            if type(node.op) in ALLOWED_OPERATORS:
                operand = safe_eval(node.operand)
                op_func = ALLOWED_OPERATORS[type(node.op)]
                return op_func(operand)
            else:
                raise ValueError("Unsupported unary operation")
                
        elif isinstance(node, ast.Compare):
            # Comparison operations
            if len(node.ops) == 1 and type(node.ops[0]) in ALLOWED_COMPARISONS:
                left = safe_eval(node.left)
                right = safe_eval(node.comparators[0])
                op_func = ALLOWED_COMPARISONS[type(node.ops[0])]
                return op_func(left, right)
            else:
                raise ValueError("Unsupported comparison")
                
        elif isinstance(node, ast.Expression):
            # Top-level expression
            return safe_eval(node.body)
            
        else:
            raise ValueError("Unsupported AST node type")
    
    try:
        # Parse the code into an AST
        tree = ast.parse(python_code, mode='eval')
        
        # Evaluate the AST safely
        result = safe_eval(tree)
        
        # Convert result to string
        return str(result)
        
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, Exception):
        return "Execution Blocked!"
