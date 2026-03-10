import re

def evaluate_expression(expression: str):
    """
    Evaluates a simple arithmetic or logical expression string.
    Handles addition, subtraction, multiplication, division,
    and logical operations 'and', 'or', 'not'.
    Returns the result of the evaluation or 'Error!' if the expression
    is invalid or unsafe.
    """
    if not isinstance(expression, str):
        return 'Error!'

    # Pattern for characters that are NOT allowed.
    # We allow numbers (0-9, .), operators (+, -, *, /), parentheses (()), whitespace (\s),
    # and letters (a-z, A-Z) for 'True', 'False', 'and', 'or', 'not'.
    # Anything else is disallowed.
    disallowed_chars_pattern = r"[^0-9\s\.\+\-\*\/\(\)a-zA-Z]"
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
        
        # Ensure the result is a number (integer or float) or a boolean.
        # Expressions like "()" would evaluate to a tuple, which is not a valid result.
        if not isinstance(result, (int, float, bool)):
            return 'Error!'
            
        return result
    except (SyntaxError, ZeroDivisionError, TypeError, NameError, OverflowError, MemoryError):
        # Catch common errors during parsing or evaluation.
        # MemoryError for computationally expensive expressions.
        return 'Error!'
    except Exception:
        # Catch any other unexpected errors.
        return 'Error!'

def evaluate_quiz_expression(quiz_expr: str):
    """
    Evaluates a user-provided Python expression for a quiz.
    The expression should only include safe operations.
    Returns the result of the evaluation or an error message
    if the input is invalid or dangerous.
    """
    if not isinstance(quiz_expr, str):
        return 'Error: Invalid input type!'

    # Pattern for characters that are NOT allowed.
    # Allows numbers, arithmetic operators, parentheses, whitespace,
    # and common keywords for simple expressions (True, False, and, or, not).
    # Adjust this pattern based on the specific needs of the quiz.
    disallowed_chars_pattern = r"[^0-9\s\.\+\-\*\/\(\)a-zA-Z]"
    if re.search(disallowed_chars_pattern, quiz_expr):
        return 'Error: Expression contains disallowed characters!'

    # Check for empty or only whitespace string.
    if not quiz_expr.strip():
        return 'Error: Expression is empty!'

    try:
        # Safely evaluate the expression.
        # The globals dict `{'__builtins__': {}}` restricts access to built-in functions.
        # The locals dict `{}` provides no local variables.
        # For a quiz, you might want to provide a limited set of safe functions or constants
        # in the locals or globals if needed, e.g., {'math': math_module_with_safe_functions_only}
        result = eval(quiz_expr, {'__builtins__': {}}, {})
        
        # Further validation on the result type might be needed depending on quiz expectations.
        # For example, if only numerical or boolean results are expected:
        # if not isinstance(result, (int, float, bool)):
        #     return 'Error: Expression resulted in an unexpected type!'
            
        return result
    except (SyntaxError, ZeroDivisionError, TypeError, NameError, OverflowError, MemoryError) as e:
        # Catch common errors during parsing or evaluation.
        return f'Error: {type(e).__name__} - Invalid or unsafe expression!'
    except Exception as e:
        # Catch any other unexpected errors.
        return f'Error: An unexpected error occurred - {type(e).__name__}!'
