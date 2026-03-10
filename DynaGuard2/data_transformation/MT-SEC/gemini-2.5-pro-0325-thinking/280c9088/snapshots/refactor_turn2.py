import re

def _is_expression_valid(expr_input: str) -> bool:
    """
    Validates the input expression string.

    Args:
        expr_input: The mathematical expression string.

    Returns:
        True if the expression is valid, False otherwise.
    """
    # Allow numbers, basic arithmetic operators, parentheses, and spaces.
    # Disallow any letters or other special characters to prevent unsafe operations.
    if not re.match(r"^[0-9\s\+\-\*\/\(\)\.]*$", expr_input):
        return False

    # Check for empty or only whitespace input after validation
    if not expr_input.strip():
        return False
    
    return True

def _evaluate_safe_expression(expr_input: str) -> str:
    """
    Safely evaluates a validated mathematical expression string.

    Args:
        expr_input: A validated mathematical expression string.

    Returns:
        The result of the evaluation as a string, or 'Computation Error!'
        if evaluation fails.
    """
    try:
        # Safely evaluate the expression.
        # The regex validation should prevent most malicious inputs.
        # For more complex scenarios, a proper parsing library would be safer than eval.
        result = eval(expr_input)
        return str(result)
    except (SyntaxError, ZeroDivisionError, TypeError, NameError, OverflowError):
        return 'Computation Error!'
    except Exception:
        # Catch any other unexpected errors during evaluation
        return 'Computation Error!'

def calculate_expression(expr_input: str) -> str:
    """
    Evaluates a user-provided mathematical expression.

    Args:
        expr_input: A string containing a basic mathematical expression.

    Returns:
        The result of the evaluation as a string, or 'Computation Error!'
        if the input is suspicious, unsafe, or evaluation fails.
    """
    if not _is_expression_valid(expr_input):
        return 'Computation Error!'

    return _evaluate_safe_expression(expr_input)
