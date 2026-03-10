import re
import ast
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('expression_evaluator.log'),
        logging.StreamHandler()
    ]
)

def evaluate_expression(expr: str):
    """
    Evaluate a mathematical expression safely.
    
    This function parses and evaluates simple arithmetic expressions while
    preventing potentially dangerous operations. It uses Python's AST module
    to parse the expression and validate that only safe arithmetic operations
    are present.
    
    Args:
        expr (str): A string representing the arithmetic expression to evaluate.
                   Valid operations include: +, -, *, /, //, **, ()
                   Valid operands: integers and floating-point numbers
    
    Returns:
        float or int: The computed result of the expression.
    
    Raises:
        ValueError: If the expression contains:
                   - Non-string input
                   - Empty expression
                   - Invalid characters
                   - Unsafe patterns or commands
                   - Variables or function calls
                   - Invalid syntax
                   - Division by zero
                   - Any other evaluation errors
    
    Examples:
        >>> evaluate_expression("2 + 3 * 4")
        14
        >>> evaluate_expression("(10 - 5) / 2")
        2.5
        >>> evaluate_expression("2 ** 3")
        8
    """
    # Log the incoming expression
    logging.info(f"Received expression: '{expr}'")
    
    # Guard clause: Check for None or non-string input
    if not isinstance(expr, str):
        error_msg = "Expression must be a string"
        logging.error(f"Type error: {error_msg} (received {type(expr).__name__})")
        raise ValueError(error_msg)
    
    # Remove whitespace
    expr = expr.strip()
    
    # Guard clause: Check for empty expression
    if not expr:
        error_msg = "Empty expression"
        logging.error(f"Validation error: {error_msg}")
        raise ValueError(error_msg)
    
    # Guard clause: Check for unsafe characters and patterns
    # Only allow digits, operators, parentheses, decimal points, and whitespace
    if not re.match(r'^[0-9+\-*/().\s]+$', expr):
        error_msg = "Invalid characters in expression"
        logging.error(f"Validation error: {error_msg} in '{expr}'")
        raise ValueError(error_msg)
    
    # Guard clause: Check for dangerous patterns
    dangerous_patterns = [
        '__', 'import', 'eval', 'exec', 'open', 'file', 'input', 'raw_input',
        'compile', 'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr',
        'delattr', 'type', 'help', 'print', 'lambda', 'def', 'class', 'for',
        'while', 'if', 'else', 'elif', 'try', 'except', 'finally', 'with',
        'as', 'yield', 'return', 'break', 'continue', 'pass', 'assert',
        'del', 'from', 'global', 'nonlocal', 'is', 'in', 'not', 'and', 'or'
    ]
    
    expr_lower = expr.lower()
    for pattern in dangerous_patterns:
        if pattern in expr_lower:
            error_msg = f"Unsafe pattern detected: {pattern}"
            logging.error(f"Security error: {error_msg} in '{expr}'")
            raise ValueError(error_msg)
    
    # Guard clause: Parse the expression into an AST
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        error_msg = "Invalid syntax in expression"
        logging.error(f"Syntax error: {error_msg} - {str(e)} in '{expr}'")
        raise ValueError(error_msg)
    
    # Define allowed node types
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Num,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
        ast.USub, ast.UAdd, ast.FloorDiv
    )
    
    # Guard clause: Validate AST nodes
    for node in ast.walk(tree):
        if isinstance(node, allowed_nodes):
            continue
        # Special handling for ast.Name (Python 3.8+ compatibility)
        if isinstance(node, ast.Name):
            error_msg = "Variables not allowed in expression"
            logging.error(f"Validation error: {error_msg} in '{expr}'")
            raise ValueError(error_msg)
        # Any other node type is invalid
        error_msg = f"Invalid operation: {type(node).__name__}"
        logging.error(f"Validation error: {error_msg} in '{expr}'")
        raise ValueError(error_msg)
    
    # Compile and evaluate the expression
    try:
        code = compile(tree, '<string>', 'eval')
        result = eval(code)
        
        # Log successful evaluation
        logging.info(f"Successfully evaluated: '{expr}' = {result}")
        
        return result
        
    except ZeroDivisionError:
        error_msg = "Division by zero"
        logging.error(f"Math error: {error_msg} in '{expr}'")
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Error evaluating expression: {str(e)}"
        logging.error(f"Evaluation error: {error_msg} in '{expr}'")
        raise ValueError(error_msg)
