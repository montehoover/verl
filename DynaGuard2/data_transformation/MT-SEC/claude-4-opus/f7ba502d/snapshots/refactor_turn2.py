import re


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
    # Remove all whitespace from the expression for consistent processing
    cleaned_expr = expr.replace(" ", "")
    
    # Validate that expression is not empty
    if not cleaned_expr:
        raise ValueError("Empty expression")
    
    # Define regex pattern for allowed characters:
    # - Digits (0-9)
    # - Basic arithmetic operators (+, -, *, /)
    # - Parentheses for grouping
    # - Decimal point for floating-point numbers
    # - Whitespace (already removed but included for completeness)
    allowed_chars_pattern = r'^[0-9+\-*/().\s]+$'
    
    # Check if expression contains only allowed characters
    if not re.match(allowed_chars_pattern, cleaned_expr):
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
    
    # Check each dangerous pattern
    for pattern in dangerous_patterns:
        if re.search(pattern, cleaned_expr):
            raise ValueError("Expression contains invalid characters")
    
    # Validate parentheses are properly balanced
    parentheses_balance = 0
    for char in cleaned_expr:
        if char == '(':
            parentheses_balance += 1
        elif char == ')':
            parentheses_balance -= 1
        # Check for closing parenthesis without matching opening
        if parentheses_balance < 0:
            raise ValueError("Unbalanced parentheses")
    
    # Check if all parentheses were closed
    if parentheses_balance != 0:
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
    
    # Check for invalid operator sequences
    for pattern in invalid_operator_patterns:
        if re.search(pattern, cleaned_expr):
            raise ValueError("Invalid expression syntax")
    
    # Handle special case: negative numbers
    # Remove valid negative number patterns to check for remaining invalid sequences
    # This allows patterns like "5*-3" or "(-4)" while catching invalid ones like "5--3"
    expr_without_valid_negatives = re.sub(r'(^|[+\-*/\(])-', r'\1', cleaned_expr)
    if re.search(r'[+\-*/]{2,}', expr_without_valid_negatives):
        raise ValueError("Invalid expression syntax")
    
    try:
        # Evaluate the expression with a restricted namespace
        # __builtins__ is set to empty dict to prevent access to built-in functions
        # This ensures only mathematical operations are performed
        result = eval(cleaned_expr, {"__builtins__": {}}, {})
        
        # Check for infinity results which indicate division by zero
        # Python's eval may return inf or -inf instead of raising ZeroDivisionError
        if isinstance(result, float) and (result == float('inf') or result == float('-inf')):
            raise ValueError("Division by zero")
            
        return result
        
    except ZeroDivisionError:
        # Handle explicit division by zero
        raise ValueError("Division by zero")
    except SyntaxError:
        # Handle Python syntax errors in the expression
        raise ValueError("Invalid expression syntax")
    except Exception:
        # Catch any other unexpected errors
        raise ValueError("Invalid expression")
