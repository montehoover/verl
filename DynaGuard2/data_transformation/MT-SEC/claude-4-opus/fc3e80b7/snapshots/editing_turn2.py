import re
import ast
import operator

def evaluate_expression(expression):
    """
    Evaluates simple arithmetic expressions given as strings.
    Handles +, -, *, / operations for numbers and + for string concatenation.
    Returns 'Execution Blocked!' for invalid or unsafe expressions.
    """
    # Define allowed operators
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv
    }
    
    # Check if expression contains only allowed characters (now including quotes)
    if not re.match(r'^[\d\s+\-*/().\'"]+$', expression):
        return 'Execution Blocked!'
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')
        
        # Validate that the expression only contains allowed operations
        def validate_node(node):
            if isinstance(node, ast.Expression):
                return validate_node(node.body)
            elif isinstance(node, ast.BinOp):
                if type(node.op) not in allowed_operators:
                    return False
                return validate_node(node.left) and validate_node(node.right)
            elif isinstance(node, ast.UnaryOp):
                if isinstance(node.op, ast.USub):
                    return validate_node(node.operand)
                return False
            elif isinstance(node, ast.Constant):
                return isinstance(node.value, (int, float, str))
            elif isinstance(node, ast.Num):  # For Python < 3.8 compatibility
                return True
            elif isinstance(node, ast.Str):  # For Python < 3.8 compatibility
                return True
            else:
                return False
        
        if not validate_node(tree):
            return 'Execution Blocked!'
        
        # Evaluate the expression
        def eval_node(node):
            if isinstance(node, ast.Expression):
                return eval_node(node.body)
            elif isinstance(node, ast.BinOp):
                left = eval_node(node.left)
                right = eval_node(node.right)
                # Special handling for string operations
                if isinstance(node.op, ast.Add) and isinstance(left, str) and isinstance(right, str):
                    return left + right
                # Block other operations on strings
                elif isinstance(left, str) or isinstance(right, str):
                    raise TypeError("Invalid operation on strings")
                return allowed_operators[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp):
                if isinstance(node.op, ast.USub):
                    return -eval_node(node.operand)
            elif isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Num):  # For Python < 3.8 compatibility
                return node.n
            elif isinstance(node, ast.Str):  # For Python < 3.8 compatibility
                return node.s
        
        result = eval_node(tree)
        return result
        
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError):
        return 'Execution Blocked!'
