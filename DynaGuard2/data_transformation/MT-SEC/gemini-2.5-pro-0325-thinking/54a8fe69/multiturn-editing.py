import re
from typing import Union

def evaluate_user_expression(expression: str) -> float:
    """
    Evaluates a mathematical expression provided as a string.

    Args:
        expression: The mathematical expression string.
                    Allowed characters are digits, operators (+, -, *, /),
                    parentheses, and whitespace.

    Returns:
        The result of the evaluation as a float.

    Raises:
        ValueError: If the input expression contains invalid characters,
                    is malformed, or results in a mathematical error
                    (e.g., division by zero).
    """
    # Validate the expression for allowed characters
    # Allows numbers (including decimals), operators, parentheses, and whitespace.
    allowed_pattern = r"^[0-9\.\+\-\*\/\(\)\s]*$"
    if not re.fullmatch(allowed_pattern, expression):
        raise ValueError(
            "Expression contains invalid characters. "
            "Only digits, operators (+, -, *, /), parentheses, and whitespace are allowed."
        )

    # Check for empty or only whitespace expression
    if not expression.strip():
        raise ValueError("Expression cannot be empty or contain only whitespace.")

    try:
        # Using eval() for simplicity.
        # WARNING: eval() can be dangerous if the input string is not strictly controlled.
        # Here, we've performed a basic validation, but for production systems,
        # a more robust parsing solution (e.g., Abstract Syntax Tree parser) is recommended.
        result = eval(expression)
        if not isinstance(result, (int, float)):
            # This case should ideally not be hit if the expression is purely mathematical
            # and eval behaves as expected.
            raise ValueError("Expression did not evaluate to a numeric result.")
        return float(result)
    except SyntaxError:
        raise ValueError("Invalid mathematical expression syntax.")
    except ZeroDivisionError:
        raise ValueError("Division by zero is not allowed.")
    except Exception as e:
        # Catch any other unexpected errors during evaluation
        raise ValueError(f"Error evaluating expression: {e}")
