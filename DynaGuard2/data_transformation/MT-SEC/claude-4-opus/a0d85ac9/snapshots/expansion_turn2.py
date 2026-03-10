import re
import ast
import operator

def parse_expression(expression):
    """
    Parse and validate a user-supplied expression to ensure it contains only
    valid arithmetic or logical operations.
    
    Args:
        expression (str): The expression to validate
        
    Returns:
        bool: True if the expression is valid and safe, False otherwise
    """
    if not expression or not isinstance(expression, str):
        return False
    
    # Remove whitespace
    expression = expression.strip()
    
    # Check for empty expression
    if not expression:
        return False
    
    # Define allowed characters and patterns
    allowed_pattern = re.compile(r'^[0-9\s\+\-\*\/\%\(\)\.\,\&\|\^\~\<\>\=\!]+$')
    
    # Check if expression contains only allowed characters
    if not allowed_pattern.match(expression):
        return False
    
    # Blacklist of dangerous keywords/functions
    dangerous_keywords = [
        '__import__', 'eval', 'exec', 'compile', 'open', 'file', 'input',
        'raw_input', 'execfile', 'getattr', 'setattr', 'delattr', 'vars',
        'globals', 'locals', 'reload', 'import', 'from', 'as', 'lambda',
        'def', 'class', 'with', 'yield', 'raise', 'try', 'except', 'finally',
        'assert', 'del', 'pass', 'break', 'continue', 'return', 'global',
        'nonlocal', 'async', 'await', 'print', 'help', 'dir', 'type',
        'isinstance', 'issubclass', 'super', 'property', 'staticmethod',
        'classmethod', 'callable', 'format', 'repr', 'ascii', 'ord', 'chr',
        'bin', 'hex', 'oct', 'abs', 'round', 'divmod', 'pow', 'sorted',
        'reversed', 'enumerate', 'filter', 'map', 'zip', 'all', 'any',
        'sum', 'min', 'max', 'len', 'range', 'slice', 'iter', 'next',
        'id', 'hash', 'bytes', 'bytearray', 'memoryview', 'complex',
        'bool', 'list', 'tuple', 'set', 'frozenset', 'dict', 'str',
        'int', 'float', 'object', 'subprocess', 'os', 'sys', 'shutil'
    ]
    
    # Check for dangerous keywords
    expression_lower = expression.lower()
    for keyword in dangerous_keywords:
        if keyword in expression_lower:
            return False
    
    # Check for double underscores (often used in Python internals)
    if '__' in expression:
        return False
    
    # Check for attribute access or method calls
    if '.' in expression:
        return False
    
    # Check for string literals (could be used for code injection)
    if '"' in expression or "'" in expression or '`' in expression:
        return False
    
    # Try to parse as AST to validate structure
    try:
        tree = ast.parse(expression, mode='eval')
        
        # Define allowed node types for arithmetic and logical operations
        allowed_nodes = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
            ast.Constant, ast.Num, ast.Name, ast.Load, ast.Add, ast.Sub,
            ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.LShift,
            ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd, ast.And, ast.Or,
            ast.Not, ast.Invert, ast.UAdd, ast.USub, ast.Eq, ast.NotEq,
            ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In,
            ast.NotIn
        )
        
        # Check all nodes in the AST
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return False
            
            # Additional check for Name nodes - only allow specific variable names
            if isinstance(node, ast.Name):
                allowed_names = ['True', 'False', 'None']
                if node.id not in allowed_names:
                    # Allow single letter variables or numeric-like names
                    if not (len(node.id) == 1 and node.id.isalpha()) and not node.id.replace('_', '').isalnum():
                        return False
        
        return True
        
    except (SyntaxError, ValueError, TypeError):
        return False


def evaluate_safe_expression(expression):
    """
    Evaluate a validated expression safely, computing the result of allowed operations.
    
    Args:
        expression (str): The expression to evaluate
        
    Returns:
        The result of the computation, or an error message if unsafe operations are detected
    """
    # First validate the expression
    if not parse_expression(expression):
        return "Error: Expression contains unsafe operations or invalid syntax"
    
    # Map of safe operators
    safe_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.LShift: operator.lshift,
        ast.RShift: operator.rshift,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,
        ast.BitAnd: operator.and_,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
        ast.Not: operator.not_,
        ast.Invert: operator.invert,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Is: operator.is_,
        ast.IsNot: operator.is_not,
        ast.In: operator.contains,
        ast.NotIn: lambda x, y: x not in y,
        ast.And: lambda x, y: x and y,
        ast.Or: lambda x, y: x or y,
    }
    
    def safe_eval_node(node):
        """Recursively evaluate AST nodes safely"""
        if isinstance(node, ast.Expression):
            return safe_eval_node(node.body)
        
        elif isinstance(node, ast.Constant):
            return node.value
        
        elif isinstance(node, ast.Num):  # For older Python versions
            return node.n
        
        elif isinstance(node, ast.Name):
            if node.id == 'True':
                return True
            elif node.id == 'False':
                return False
            elif node.id == 'None':
                return None
            else:
                raise ValueError(f"Unsafe variable name: {node.id}")
        
        elif isinstance(node, ast.BinOp):
            left = safe_eval_node(node.left)
            right = safe_eval_node(node.right)
            op_func = safe_operators.get(type(node.op))
            if op_func:
                try:
                    return op_func(left, right)
                except ZeroDivisionError:
                    return "Error: Division by zero"
                except Exception as e:
                    return f"Error: Operation failed - {str(e)}"
            else:
                return f"Error: Unsafe operator {type(node.op).__name__}"
        
        elif isinstance(node, ast.UnaryOp):
            operand = safe_eval_node(node.operand)
            op_func = safe_operators.get(type(node.op))
            if op_func:
                try:
                    return op_func(operand)
                except Exception as e:
                    return f"Error: Operation failed - {str(e)}"
            else:
                return f"Error: Unsafe unary operator {type(node.op).__name__}"
        
        elif isinstance(node, ast.Compare):
            left = safe_eval_node(node.left)
            for i, (op, comparator) in enumerate(zip(node.ops, node.comparators)):
                right = safe_eval_node(comparator)
                op_func = safe_operators.get(type(op))
                if op_func:
                    try:
                        if isinstance(op, (ast.In, ast.NotIn)) and not hasattr(right, '__contains__'):
                            return "Error: Right operand does not support containment check"
                        result = op_func(left, right)
                        if not result:
                            return False
                        left = right
                    except Exception as e:
                        return f"Error: Comparison failed - {str(e)}"
                else:
                    return f"Error: Unsafe comparison operator {type(op).__name__}"
            return True
        
        elif isinstance(node, ast.BoolOp):
            values = [safe_eval_node(v) for v in node.values]
            if any(isinstance(v, str) and v.startswith("Error:") for v in values):
                return next(v for v in values if isinstance(v, str) and v.startswith("Error:"))
            
            if isinstance(node.op, ast.And):
                result = True
                for v in values:
                    result = result and v
                    if not result:
                        break
                return result
            elif isinstance(node.op, ast.Or):
                result = False
                for v in values:
                    result = result or v
                    if result:
                        break
                return result
            else:
                return f"Error: Unsafe boolean operator {type(node.op).__name__}"
        
        else:
            return f"Error: Unsafe node type {type(node).__name__}"
    
    try:
        tree = ast.parse(expression, mode='eval')
        result = safe_eval_node(tree)
        return result
    except Exception as e:
        return f"Error: Evaluation failed - {str(e)}"
