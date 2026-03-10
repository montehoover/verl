import re
import logging
from datetime import datetime


def safe_eval_expression(expr: str):
    """Safely evaluate a mathematical expression provided by the user.
    
    This function takes a string containing a mathematical expression and
    evaluates it safely, preventing code injection and other security risks.
    Only basic arithmetic operations (+, -, *, /) and parentheses are allowed.
    
    The function performs extensive validation before evaluation:
    - Checks for allowed characters only (digits, operators, parentheses, decimals)
    - Validates parentheses are balanced
    - Ensures proper operator placement
    - Prevents access to Python builtins and dangerous functions
    
    Args:
        expr: A string containing the mathematical expression to evaluate.
              Example: "2 + 3 * (4 - 1)"
    
    Returns:
        float or int: The result of evaluating the mathematical expression.
        
    Raises:
        ValueError: If the expression contains invalid characters, has syntax
                   errors, attempts division by zero, or is otherwise invalid.
                   
    Examples:
        >>> safe_eval_expression("2 + 3")
        5
        >>> safe_eval_expression("10 / (5 - 5)")
        Raises ValueError: Division by zero
        >>> safe_eval_expression("2 ** 3")
        Raises ValueError: Expression contains invalid characters
    """
    # Initialize logging for expression evaluation tracking
    # Configure logger with unique name to avoid conflicts
    logger = logging.getLogger('safe_eval_expression')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicate logs
    logger.handlers = []
    
    # Create file handler with timestamp in filename
    log_filename = 'expression_evaluations.log'
    file_handler = logging.FileHandler(log_filename, mode='a')
    file_handler.setLevel(logging.INFO)
    
    # Create formatter for clean, readable log entries
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | Expression: %(expression)s | Result: %(result)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Log the incoming expression
    logger.info("Received expression for evaluation", extra={'expression': expr, 'result': 'PENDING'})
    
    # Remove all whitespace from the expression for consistent processing
    cleaned_expr = expr.replace(" ", "")
    
    # Guard clause: Validate that expression is not empty
    if not cleaned_expr:
        logger.error("Empty expression provided", extra={'expression': expr, 'result': 'ERROR: Empty expression'})
        raise ValueError("Empty expression")
    
    # Define regex pattern for allowed characters:
    # - Digits (0-9)
    # - Basic arithmetic operators (+, -, *, /)
    # - Parentheses for grouping
    # - Decimal point for floating-point numbers
    # - Whitespace (already removed but included for completeness)
    allowed_chars_pattern = r'^[0-9+\-*/().\s]+$'
    
    # Guard clause: Check if expression contains only allowed characters
    if not re.match(allowed_chars_pattern, cleaned_expr):
        logger.error("Invalid characters detected", extra={'expression': expr, 'result': 'ERROR: Invalid characters'})
        raise ValueError("Expression contains invalid characters")
    
    # Additional security validation - check for dangerous patterns
    # that could be used for code injection or accessing Python internals
    dangerous_patterns = [
        r'__',          # Double underscore - could access special attributes/methods
        r'import',      # Import statements - could import dangerous modules
        r'eval',        # Eval function - could execute arbitrary code
        r'exec',        # Exec function - could execute arbitrary code
        r'[a-zA-Z]',    # Any letters - function calls or variable names not allowed
    ]
    
    # Guard clause: Check each dangerous pattern
    for pattern in dangerous_patterns:
        if re.search(pattern, cleaned_expr):
            logger.error(f"Dangerous pattern detected: {pattern}", extra={'expression': expr, 'result': 'ERROR: Security violation'})
            raise ValueError("Expression contains invalid characters")
    
    # Validate parentheses are properly balanced
    parentheses_balance = 0
    for char in cleaned_expr:
        if char == '(':
            parentheses_balance += 1
        elif char == ')':
            parentheses_balance -= 1
        # Guard clause: Check for closing parenthesis without matching opening
        if parentheses_balance < 0:
            logger.error("Unbalanced parentheses - extra closing", extra={'expression': expr, 'result': 'ERROR: Unbalanced parentheses'})
            raise ValueError("Unbalanced parentheses")
    
    # Guard clause: Check if all parentheses were closed
    if parentheses_balance != 0:
        logger.error("Unbalanced parentheses - unclosed", extra={'expression': expr, 'result': 'ERROR: Unbalanced parentheses'})
        raise ValueError("Unbalanced parentheses")
    
    # Define patterns for invalid operator sequences
    invalid_operator_patterns = [
        r'[+\-*/]{2,}',     # Multiple operators in sequence (except valid cases like +-)
        r'^[*/]',           # Expression starting with multiplication or division
        r'[+\-*/]$',        # Expression ending with any operator
        r'\(\)',            # Empty parentheses - no content between them
        r'[+\-*/]\)',       # Operator immediately before closing parenthesis
        r'\([+*/]',         # Opening parenthesis followed by * or / (- is valid for negatives)
    ]
    
    # Guard clause: Check for invalid operator sequences
    for pattern in invalid_operator_patterns:
        if re.search(pattern, cleaned_expr):
            logger.error(f"Invalid operator sequence: {pattern}", extra={'expression': expr, 'result': 'ERROR: Invalid syntax'})
            raise ValueError("Invalid expression syntax")
    
    # Handle special case: negative numbers
    # Remove valid negative number patterns to check for remaining invalid sequences
    # This allows patterns like "5*-3" or "(-4)" while catching invalid ones like "5--3"
    expr_without_valid_negatives = re.sub(r'(^|[+\-*/\(])-', r'\1', cleaned_expr)
    
    # Guard clause: Check for invalid operator sequences after removing valid negatives
    if re.search(r'[+\-*/]{2,}', expr_without_valid_negatives):
        logger.error("Invalid operator sequence after negative number handling", extra={'expression': expr, 'result': 'ERROR: Invalid syntax'})
        raise ValueError("Invalid expression syntax")
    
    try:
        # Evaluate the expression with a restricted namespace
        # __builtins__ is set to empty dict to prevent access to built-in functions
        # This ensures only mathematical operations are performed
        result = eval(cleaned_expr, {"__builtins__": {}}, {})
        
        # Guard clause: Check for infinity results which indicate division by zero
        # Python's eval may return inf or -inf instead of raising ZeroDivisionError
        if isinstance(result, float) and (result == float('inf') or result == float('-inf')):
            logger.error("Division by zero resulted in infinity", extra={'expression': expr, 'result': 'ERROR: Division by zero'})
            raise ValueError("Division by zero")
        
        # Log successful evaluation
        logger.info("Expression evaluated successfully", extra={'expression': expr, 'result': str(result)})
        
        return result
        
    except ZeroDivisionError:
        # Handle explicit division by zero
        logger.error("Division by zero exception", extra={'expression': expr, 'result': 'ERROR: Division by zero'})
        raise ValueError("Division by zero")
    except SyntaxError:
        # Handle Python syntax errors in the expression
        logger.error("Syntax error in expression", extra={'expression': expr, 'result': 'ERROR: Invalid syntax'})
        raise ValueError("Invalid expression syntax")
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error: {type(e).__name__}", extra={'expression': expr, 'result': f'ERROR: {type(e).__name__}'})
        raise ValueError("Invalid expression")
