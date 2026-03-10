import ast

def _validate_expression_syntax(calc_string: str) -> bool:
    """
    Validates if the given string is a syntactically valid Python expression.
    An empty string or a string with only whitespace is considered invalid.
    """
    if not isinstance(calc_string, str) or not calc_string.strip():
        return False
    try:
        # Use mode='eval' to ensure it's a single expression
        ast.parse(calc_string, mode='eval')
        return True
    except SyntaxError:
        return False

def _evaluate_validated_expression(calc_string: str) -> str:
    """
    Evaluates a syntactically validated expression string in a restricted environment.
    
    This function expects calc_string to have passed _validate_expression_syntax.
    It may raise various exceptions during evaluation (e.g., ZeroDivisionError, 
    NameError for undefined variables, TypeError for operations on incompatible types),
    which are expected to be caught by the caller.
    """
    # Evaluate the expression in a highly restricted environment.
    # Only basic arithmetic should work. Access to builtins is denied.
    result = eval(calc_string, {"__builtins__": {}}, {})
    return str(result)

def exec_calculation(calc_string: str) -> str:
    """
    Processes numerical expressions provided by users as Python code and evaluates the result.

    Args:
        calc_string: A user-provided string that should contain only basic
                     mathematical expressions to evaluate.

    Returns:
        The result of the evaluated expression as a string, or 'Computation Error!'
        in case of failure or if harmful code is detected.
    """
    if not _validate_expression_syntax(calc_string):
        return 'Computation Error!'

    try:
        # _evaluate_validated_expression can raise various exceptions (e.g., ZeroDivisionError, NameError).
        # These are caught here and uniformly reported as 'Computation Error!'.
        result_str = _evaluate_validated_expression(calc_string)
        return result_str
    except Exception: # Catches any exception from _evaluate_validated_expression or other unexpected errors
        return 'Computation Error!'
