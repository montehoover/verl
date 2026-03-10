import re
import ast
import operator

def validate_math_expression(expression):
    """
    Validates a mathematical expression to ensure it contains only safe mathematical operations.
    
    Args:
        expression (str): The mathematical expression to validate
        
    Returns:
        bool: True if the expression is valid and safe, False otherwise
    """
    if not expression or not isinstance(expression, str):
        return False
    
    # Remove whitespace
    expression = expression.strip()
    
    if not expression:
        return False
    
    # Check for forbidden characters and patterns
    forbidden_patterns = [
        '__',  # Double underscore (dunder methods)
        'import',
        'exec',
        'eval',
        'compile',
        'open',
        'file',
        'input',
        'raw_input',
        'globals',
        'locals',
        'vars',
        'dir',
        'getattr',
        'setattr',
        'delattr',
        'hasattr',
        'callable',
        'classmethod',
        'staticmethod',
        'property',
        'super',
        'type',
        'isinstance',
        'issubclass',
        'lambda',
        'def',
        'class',
        'for',
        'while',
        'if',
        'else',
        'elif',
        'try',
        'except',
        'finally',
        'raise',
        'assert',
        'with',
        'as',
        'yield',
        'from',
        'return',
        'break',
        'continue',
        'pass',
        'del',
        'is',
        'in',
        'not',
        'and',
        'or',
        ';',  # Statement separator
        '\\',  # Escape character
        '`',  # Backtick
        '$',  # Dollar sign
        '{',  # Curly braces
        '}',
        '[',  # Square brackets
        ']',
    ]
    
    expression_lower = expression.lower()
    for pattern in forbidden_patterns:
        if pattern in expression_lower:
            return False
    
    # Allow only specific characters: digits, operators, parentheses, decimal points, and whitespace
    allowed_pattern = r'^[0-9+\-*/().\s]+$'
    if not re.match(allowed_pattern, expression):
        return False
    
    # Check for balanced parentheses
    paren_count = 0
    for char in expression:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        if paren_count < 0:
            return False
    if paren_count != 0:
        return False
    
    # Try to parse as an AST to ensure it's a valid expression
    try:
        tree = ast.parse(expression, mode='eval')
        
        # Whitelist of allowed AST node types
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Num,  # For Python < 3.8
            ast.Constant,  # For Python >= 3.8
        )
        
        # Check all nodes in the AST
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return False
                
        return True
        
    except (SyntaxError, ValueError, TypeError):
        return False


def compute_safe_expression(expression):
    """
    Computes the result of a mathematical expression safely without using eval() or exec().
    
    Args:
        expression (str): The mathematical expression to compute
        
    Returns:
        float or str: The result of the computation, or an error message if there's a safety violation
    """
    # First validate the expression
    if not validate_math_expression(expression):
        return "Error: Invalid or unsafe expression detected"
    
    # Map of operators
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    def evaluate_node(node):
        """Recursively evaluate an AST node"""
        if isinstance(node, ast.Expression):
            return evaluate_node(node.body)
        
        elif isinstance(node, ast.Num):  # Python < 3.8
            return float(node.n)
        
        elif isinstance(node, ast.Constant):  # Python >= 3.8
            if isinstance(node.value, (int, float)):
                return float(node.value)
            else:
                raise ValueError("Non-numeric constant detected")
        
        elif isinstance(node, ast.BinOp):
            left = evaluate_node(node.left)
            right = evaluate_node(node.right)
            op_func = ops.get(type(node.op))
            if op_func is None:
                raise ValueError("Unsupported operation detected")
            
            # Check for division by zero
            if isinstance(node.op, ast.Div) and right == 0:
                return "Error: Division by zero"
            
            try:
                result = op_func(left, right)
                # Check for extremely large numbers that might cause issues
                if abs(result) > 1e100:
                    return "Error: Result too large"
                return result
            except (OverflowError, ValueError):
                return "Error: Arithmetic overflow or invalid operation"
        
        elif isinstance(node, ast.UnaryOp):
            operand = evaluate_node(node.operand)
            op_func = ops.get(type(node.op))
            if op_func is None:
                raise ValueError("Unsupported unary operation detected")
            return op_func(operand)
        
        else:
            raise ValueError("Unsupported node type detected")
    
    try:
        # Parse the expression
        tree = ast.parse(expression.strip(), mode='eval')
        
        # Evaluate the expression tree
        result = evaluate_node(tree)
        
        # Return the result
        return result
        
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError) as e:
        return f"Error: Potential safety violation - {str(e)}"
    except Exception:
        return "Error: Unexpected error during computation"
