import re

def evaluate_expression(expression: str):
    """
    Evaluates a simple arithmetic expression string.
    Handles addition, subtraction, multiplication, and division.
    Returns the result of the evaluation or 'Error!' if the expression
    is invalid or unsafe.
    """
    if not isinstance(expression, str):
        return 'Error!'

    # Pattern for characters that are NOT allowed.
    # We allow numbers (0-9, .), operators (+, -, *, /), parentheses (()), and whitespace (\s).
    # Anything else is disallowed.
    disallowed_chars_pattern = r"[^0-9\s\.\+\-\*\/\(\)]"
    if re.search(disallowed_chars_pattern, expression):
        return 'Error!'

    # Check for empty or only whitespace string.
    if not expression.strip():
        return 'Error!'

    try:
        # Safely evaluate the expression.
        # The globals dict `{'__builtins__': {}}` restricts access to built-in functions.
        # The locals dict `{}` provides no local variables.
        # WARNING: eval() can still be risky with untrusted input, e.g., complex expressions
        # leading to resource exhaustion. For production systems with untrusted input,
        # a proper parser (e.g., using the ast module) is highly recommended.
        result = eval(expression, {'__builtins__': {}}, {})
        
        # Ensure the result is a number (integer or float).
        # Expressions like "()" would evaluate to a tuple, which is not a numeric result.
        if not isinstance(result, (int, float)):
            return 'Error!'
            
        return result
    except (SyntaxError, ZeroDivisionError, TypeError, NameError, OverflowError, MemoryError):
        # Catch common errors during parsing or evaluation.
        # MemoryError for computationally expensive expressions.
        return 'Error!'
    except Exception:
        # Catch any other unexpected errors.
        return 'Error!'
