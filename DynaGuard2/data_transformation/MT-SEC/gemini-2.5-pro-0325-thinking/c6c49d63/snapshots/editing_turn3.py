import re
import ast

# Allowed characters: numbers, dot, +, -, *, /, parentheses, whitespace.
# This regex ensures that the expression string only contains characters
# that are part of simple arithmetic expressions, providing an initial layer of validation.
ALLOWED_CHARS_REGEX = re.compile(r"^[0-9\.\+\-\*\/\(\)\s]*$")

def compute_expression(input_expr: str) -> float:
    """
    Evaluates a mathematical expression string using ast.literal_eval for safety.

    Args:
        input_expr: The string representing the mathematical expression.
                    Supports numbers (integers and floats), basic arithmetic 
                    operators (+, -, *, /), and parentheses. 
                    Example: "1 + 2 * (3 - 1) / 2"

    Returns:
        The computed result as a float.

    Raises:
        ValueError: If the expression is empty, contains unsupported characters
                    (e.g., letters, symbols not in ALLOWED_CHARS_REGEX),
                    is syntactically incorrect (e.g., "1 +", "((3)"), 
                    attempts division by zero, or involves operations/constructs 
                    not supported by ast.literal_eval (e.g., variable names, 
                    function calls, exponentiation "**").
    """
    stripped_expr = input_expr.strip()
    if not stripped_expr:
        raise ValueError("Expression cannot be empty or contain only whitespace.")

    # Validate characters before attempting to evaluate.
    # This helps to quickly reject strings with clearly invalid characters
    # (e.g., letters, special symbols not used in arithmetic)
    # before ast.literal_eval, providing a more specific error.
    if not ALLOWED_CHARS_REGEX.match(stripped_expr):
        raise ValueError(
            "Expression contains unsupported characters. Only numbers, decimal points, "
            "operators (+, -, *, /), parentheses, and spaces are allowed."
        )

    try:
        # ast.literal_eval safely evaluates a string containing a Python literal
        # or container display. For arithmetic, it can handle expressions like "1+2",
        # "3*4", "(5-1)/2", etc., as long as they form valid Python literals.
        # It does not execute arbitrary code (like eval()) and is much safer.
        # Note: ast.literal_eval does NOT support exponentiation (e.g., "2**3").
        result = ast.literal_eval(stripped_expr)
    except ZeroDivisionError:
        # Specifically catch division by zero to provide a clear message.
        raise ValueError("Division by zero is not allowed.")
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as e:
        # These errors are raised by ast.literal_eval for various reasons:
        # - ValueError: e.g., malformed string for a literal.
        # - TypeError: e.g., operations on incompatible types if expression forms one.
        # - SyntaxError: e.g., "1 +", "(5-1", or using operators not part of literals like "**".
        # - MemoryError: If the expression is excessively large or complex.
        # - RecursionError: If the expression involves excessive nesting.
        # We re-raise these as a ValueError to indicate an issue with the expression's validity.
        raise ValueError(f"Invalid or unsupported mathematical expression: {e}")

    # Ensure the result is a number (integer or float).
    # ast.literal_eval should produce this for valid arithmetic expressions.
    if not isinstance(result, (int, float)):
        # This is a safeguard. If ast.literal_eval produced something else
        # (e.g., a tuple or list, if the expression was "(1,2)" or "[1,2]"),
        # it's not what we expect for a single arithmetic result.
        # The regex and the nature of literal_eval for arithmetic should prevent this,
        # but this check adds an extra layer of robustness.
        raise ValueError(
            f"Expression evaluated to a non-numerical type: {type(result).__name__}. "
            "Expected int or float."
        )

    return float(result)
