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


def safely_execute_expression(expression_code):
    """
    Safely execute a Python expression and return the result.
    
    Args:
        expression_code (str): A string containing a Python expression
        
    Returns:
        Any or str: The result of the expression or an error message
    """
    # Check for dangerous keywords and patterns
    dangerous_keywords = [
        '__import__', 'import', 'eval', 'exec', 'compile', 'open', 'file', 
        'input', 'raw_input', 'execfile', 'reload', 'globals', 'locals',
        'vars', 'dir', 'getattr', 'setattr', 'delattr', 'hasattr',
        '__builtins__', '__dict__', '__class__', '__bases__', '__subclasses__',
        'type', 'help', 'memoryview', 'bytearray', '__code__', '__closure__',
        'func_', 'lambda', 'def', 'class', 'for', 'while', 'if', 'elif',
        'else', 'try', 'except', 'finally', 'with', 'as', 'yield', 'from',
        'raise', 'del', 'pass', 'break', 'continue', 'return', 'global',
        'nonlocal', 'assert', 'async', 'await', 'print'
    ]
    
    expression_lower = expression_code.lower()
    for keyword in dangerous_keywords:
        if keyword in expression_lower:
            return "Error: Unsafe operation detected"
    
    # Check for double underscores
    if '__' in expression_code:
        return "Error: Unsafe operation detected"
    
    # Check for dot notation (attribute access)
    if '.' in expression_code:
        return "Error: Unsafe operation detected"
    
    # Check for square brackets (indexing/slicing could be exploited)
    if '[' in expression_code or ']' in expression_code:
        return "Error: Unsafe operation detected"
    
    try:
        # Parse and validate the AST
        tree = ast.parse(expression_code, mode='eval')
        
        # Define allowed node types
        allowed_node_types = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare,
            ast.Num, ast.Constant, ast.Name, ast.Load,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
            ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
            ast.Is, ast.IsNot, ast.In, ast.NotIn, ast.BoolOp
        )
        
        # Walk through AST and check all nodes
        for node in ast.walk(tree):
            if not isinstance(node, allowed_node_types):
                return "Error: Unsafe operation detected"
            
            # Additional check for Name nodes
            if isinstance(node, ast.Name):
                # Only allow specific safe names
                allowed_names = {'True', 'False', 'None'}
                if node.id not in allowed_names:
                    # Allow simple variable names for numeric literals
                    if not node.id.replace('_', '').isalnum():
                        return "Error: Unsafe operation detected"
        
        # Create a restricted namespace
        safe_dict = {
            'True': True,
            'False': False,
            'None': None,
        }
        
        # Evaluate the expression in the restricted namespace
        result = eval(expression_code, {"__builtins__": {}}, safe_dict)
        return result
        
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"
