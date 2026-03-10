import re
import ast
import operator

def calculate_expression(expression):
    """
    Evaluate a simple arithmetic expression given as a string.
    
    Args:
        expression (str): A string containing a simple arithmetic expression
        
    Returns:
        float or str: The result of the arithmetic expression or error message
    """
    # Remove whitespace
    expression = expression.replace(" ", "")
    
    # Check if expression contains only allowed characters
    if not re.match(r'^[0-9+\-*/().\s]+$', expression):
        return "Unsafe Operation Detected!"
    
    # Check for common unsafe patterns
    unsafe_patterns = ['__', 'import', 'eval', 'exec', 'open', 'file', 'input', 'compile']
    for pattern in unsafe_patterns:
        if pattern in expression.lower():
            return "Unsafe Operation Detected!"
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')
        
        # Define allowed operations
        allowed_operations = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }
        
        def evaluate_node(node):
            if isinstance(node, ast.Expression):
                return evaluate_node(node.body)
            elif isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Num):  # For Python < 3.8 compatibility
                return node.n
            elif isinstance(node, ast.BinOp):
                left = evaluate_node(node.left)
                right = evaluate_node(node.right)
                op_type = type(node.op)
                if op_type in allowed_operations:
                    return allowed_operations[op_type](left, right)
                else:
                    raise ValueError("Unsafe operation")
            elif isinstance(node, ast.UnaryOp):
                operand = evaluate_node(node.operand)
                op_type = type(node.op)
                if op_type in allowed_operations:
                    return allowed_operations[op_type](operand)
                else:
                    raise ValueError("Unsafe operation")
            else:
                raise ValueError("Unsafe operation")
        
        result = evaluate_node(tree)
        return float(result)
        
    except (ValueError, ZeroDivisionError, SyntaxError, TypeError):
        return "Unsafe Operation Detected!"
