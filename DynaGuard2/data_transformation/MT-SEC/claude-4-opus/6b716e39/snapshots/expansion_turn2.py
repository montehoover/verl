import re
import ast
import operator

def validate_expression(expression):
    """
    Validates if a string contains only valid arithmetic or logical operations.
    Returns True if valid, False otherwise.
    """
    # Check for empty or None input
    if not expression or not isinstance(expression, str):
        return False
    
    # Remove whitespace for analysis
    expression = expression.strip()
    
    # Empty expression after stripping
    if not expression:
        return False
    
    # Define allowed characters and patterns
    # Numbers, operators, parentheses, and common math functions
    allowed_pattern = r'^[0-9\+\-\*/\(\)\s\.\,\^\%\&\|\!\=\<\>]+$'
    
    # Check for dangerous keywords that could indicate code injection
    dangerous_keywords = [
        'import', 'exec', 'eval', '__', 'open', 'file', 'input',
        'compile', 'globals', 'locals', 'vars', 'dir', 'help',
        'type', 'isinstance', 'getattr', 'setattr', 'delattr',
        'hasattr', 'callable', 'classmethod', 'staticmethod',
        'property', 'super', 'object', 'print', 'exit', 'quit',
        'os', 'sys', 'subprocess', 'pickle', 'shelve', 'marshal',
        'importlib', 'runpy', 'execfile', 'raw_input', 'reload',
        'del', 'lambda', 'def', 'class', 'with', 'as', 'yield',
        'return', 'break', 'continue', 'pass', 'raise', 'try',
        'except', 'finally', 'assert', 'global', 'nonlocal',
        'from', 'import', 'for', 'while', 'if', 'elif', 'else'
    ]
    
    # Convert to lowercase for case-insensitive checking
    expression_lower = expression.lower()
    
    # Check for dangerous keywords
    for keyword in dangerous_keywords:
        if keyword in expression_lower:
            return False
    
    # Check if expression matches allowed pattern
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
    
    # Additional check for consecutive operators (except for negative numbers)
    operator_pattern = r'[\+\-\*/\^%]{2,}'
    if re.search(operator_pattern, expression):
        # Allow -- for double negative, but not other combinations
        cleaned = re.sub(r'--', '', expression)
        if re.search(operator_pattern, cleaned):
            return False
    
    return True


def evaluate_valid_expression(expression):
    """
    Safely evaluates a validated expression and returns the result as a string.
    Returns an error message if evaluation fails or expression is unsafe.
    """
    # First validate the expression
    if not validate_expression(expression):
        return "Evaluation failed: Expression contains unauthorized operations"
    
    try:
        # Replace ^ with ** for exponentiation
        expression = expression.replace('^', '**')
        
        # Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')
        
        # Define allowed node types
        allowed_nodes = {
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
            ast.USub, ast.UAdd, ast.FloorDiv, ast.Compare, ast.BoolOp,
            ast.And, ast.Or, ast.Not, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
            ast.Gt, ast.GtE, ast.BitAnd, ast.BitOr, ast.BitXor
        }
        
        # Check if all nodes in the AST are allowed
        for node in ast.walk(tree):
            if type(node) not in allowed_nodes:
                # Handle compatibility for different Python versions
                if hasattr(ast, 'NameConstant') and isinstance(node, ast.NameConstant):
                    continue
                if isinstance(node, ast.Name) and node.id in ('True', 'False'):
                    continue
                return "Evaluation failed: Expression contains unauthorized operations"
        
        # Compile and evaluate the expression
        compiled = compile(tree, '<string>', 'eval')
        
        # Create a restricted environment
        safe_dict = {
            '__builtins__': {},
            'True': True,
            'False': False
        }
        
        # Evaluate the expression
        result = eval(compiled, safe_dict)
        
        # Convert result to string
        return str(result)
        
    except ZeroDivisionError:
        return "Evaluation failed: Division by zero"
    except OverflowError:
        return "Evaluation failed: Result too large"
    except ValueError:
        return "Evaluation failed: Invalid mathematical operation"
    except SyntaxError:
        return "Evaluation failed: Invalid expression syntax"
    except Exception:
        return "Evaluation failed: Expression could not be evaluated"
