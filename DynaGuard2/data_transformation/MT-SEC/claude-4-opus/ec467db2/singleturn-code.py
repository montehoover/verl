import ast
import operator

def execute_user_code(user_code: str) -> str:
    """
    Securely parse and execute a Python script containing basic arithmetic or string operations.
    
    Args:
        user_code: str - A Python script submitted by the user containing simple arithmetic or string manipulations
        
    Returns:
        str - The evaluated result of the script, or 'Execution Blocked!' if the script contains dangerous actions
    """
    
    # Define safe operations
    safe_operators = {
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
    
    safe_comparisons = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }
    
    def evaluate_node(node):
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Constant):
            # Handle literals (numbers, strings, etc.)
            return node.value
        
        elif isinstance(node, ast.BinOp):
            # Handle binary operations
            if type(node.op) not in safe_operators:
                raise ValueError("Unsafe operation")
            left = evaluate_node(node.left)
            right = evaluate_node(node.right)
            return safe_operators[type(node.op)](left, right)
        
        elif isinstance(node, ast.UnaryOp):
            # Handle unary operations
            if type(node.op) not in safe_operators:
                raise ValueError("Unsafe operation")
            operand = evaluate_node(node.operand)
            return safe_operators[type(node.op)](operand)
        
        elif isinstance(node, ast.Compare):
            # Handle comparisons
            left = evaluate_node(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                if type(op) not in safe_comparisons:
                    raise ValueError("Unsafe operation")
                right = evaluate_node(comparator)
                if not safe_comparisons[type(op)](left, right):
                    return False
                left = right
            return True
        
        elif isinstance(node, ast.BoolOp):
            # Handle boolean operations (and, or)
            if isinstance(node.op, ast.And):
                for value in node.values:
                    if not evaluate_node(value):
                        return False
                return True
            elif isinstance(node.op, ast.Or):
                for value in node.values:
                    if evaluate_node(value):
                        return True
                return False
            else:
                raise ValueError("Unsafe operation")
        
        elif isinstance(node, ast.IfExp):
            # Handle conditional expressions (ternary operator)
            test = evaluate_node(node.test)
            if test:
                return evaluate_node(node.body)
            else:
                return evaluate_node(node.orelse)
        
        else:
            # Any other node type is considered unsafe
            raise ValueError("Unsafe operation")
    
    try:
        # Parse the user code
        tree = ast.parse(user_code, mode='eval')
        
        # Check if it's a simple expression
        if not isinstance(tree, ast.Expression):
            return "Execution Blocked!"
        
        # Evaluate the expression
        result = evaluate_node(tree.body)
        
        # Convert result to string
        return str(result)
        
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, Exception):
        # Any error results in execution being blocked
        return "Execution Blocked!"
