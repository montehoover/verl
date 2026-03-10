import re

def evaluate_user_expression(expression: str):
    """
    Safely calculates the result of a mathematical expression provided by the user.

    Args:
        expression: str, the string containing the user’s mathematical expression.

    Returns:
        The result of evaluating the mathematical expression.

    Raises:
        ValueError: if the input contains invalid characters or if the expression is incorrect.
    """
    # Define a pattern for allowed characters in a mathematical expression:
    # - digits (0-9)
    # - whitespace (\s)
    # - decimal point (\.)
    # - parentheses (\( \))
    # - basic arithmetic operators (\+, \-, \*, \/)
    allowed_chars_pattern = r"^[0-9\s\.\(\)\+\-\*\/]*$"

    if not re.fullmatch(allowed_chars_pattern, expression):
        raise ValueError("Invalid characters in expression.")

    # If the expression is empty or only whitespace after validation, eval will raise SyntaxError.
    if not expression.strip():
        raise ValueError("Expression cannot be empty or just whitespace.")

    try:
        # Safely evaluate the expression.
        # Restrict globals to prevent access to built-in functions or other potentially unsafe operations.
        # An empty dictionary for locals is also good practice.
        result = eval(expression, {"__builtins__": {}}, {})
        return result
    except SyntaxError:
        raise ValueError("Invalid expression syntax.")
    except ZeroDivisionError:
        raise ValueError("Division by zero in expression.")
    except Exception as e:
        # Catch any other errors during evaluation (e.g., TypeError, NameError if somehow bypassed)
        # and wrap them in a ValueError.
        raise ValueError(f"Error evaluating expression: {str(e)}")
