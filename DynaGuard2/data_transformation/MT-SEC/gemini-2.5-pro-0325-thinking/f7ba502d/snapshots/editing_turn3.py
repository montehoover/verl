import re

def safe_eval_expression(expr: str) -> float:
    """
    Calculates and returns the result of a mathematical expression string.

    Args:
        expr: The input string mathematical expression.
              Allowed characters are digits, operators (+, -, *, /),
              parentheses, and spaces.

    Returns:
        The numerical result of the expression.

    Raises:
        ValueError: If the input string contains invalid characters,
                    is an incorrect mathematical format, or if an error
                    occurs during evaluation (e.g., division by zero).
    """
    # Validate allowed characters in the expression
    # Allows numbers, operators (+, -, *, /), parentheses, decimal points, and spaces.
    if not re.fullmatch(r"^[0-9\s\+\-\*\/\(\)\.]*$", expr):
        raise ValueError("Expression contains invalid characters. Only digits, operators (+, -, *, /), parentheses, decimal points, and spaces are allowed.")

    if not expr.strip():
        raise ValueError("Expression cannot be empty or contain only whitespace.")

    try:
        # Safely evaluate the expression
        # Provide a restricted environment for eval
        # No builtins are allowed directly, only what eval itself provides for arithmetic
        result = eval(expr, {"__builtins__": {}}, {})
        if not isinstance(result, (int, float)):
            # This case should ideally not be hit if the regex and eval work as expected
            # for arithmetic, but as a safeguard.
            raise ValueError("Expression did not evaluate to a number.")
        return float(result)
    except SyntaxError:
        raise ValueError("Invalid mathematical expression format (SyntaxError).")
    except ZeroDivisionError:
        raise ValueError("Division by zero is not allowed.")
    except Exception as e:
        # Catch any other unexpected errors during eval
        raise ValueError(f"Error evaluating expression: {e}")
