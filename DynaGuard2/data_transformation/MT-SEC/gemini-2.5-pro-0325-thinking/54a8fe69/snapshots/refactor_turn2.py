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
    # The pattern ensures that only these characters are present in the string from start (^) to end ($).
    allowed_chars_pattern = r"^[0-9\s\.\(\)\+\-\*\/]*$"

    # Validate the entire expression against the allowed characters.
    if not re.fullmatch(allowed_chars_pattern, expression):
        raise ValueError("Invalid characters in expression. Only numbers, operators (+, -, *, /), parentheses, and spaces are allowed.")

    # Check if the expression is empty or contains only whitespace after initial validation.
    # This prevents eval() from raising a SyntaxError on an empty string.
    if not expression.strip():
        raise ValueError("Expression cannot be empty or just whitespace.")

    try:
        # Evaluate the expression using eval().
        # To mitigate security risks associated with eval(), we restrict its environment:
        # 1. `{"__builtins__": {}}`: This provides an empty dictionary for the `globals` argument,
        #    effectively removing access to all built-in functions and names.
        # 2. `{}`: This provides an empty dictionary for the `locals` argument.
        # This setup significantly limits what the evaluated expression can do,
        # preventing execution of arbitrary code or access to sensitive information.
        result = eval(expression, {"__builtins__": {}}, {})
        return result
    except SyntaxError:
        # Handle cases where the expression is syntactically incorrect (e.g., "1 +", "((2*3)").
        raise ValueError("Invalid expression syntax.")
    except ZeroDivisionError:
        # Handle division by zero errors explicitly.
        raise ValueError("Division by zero in expression.")
    except Exception as e:
        # Catch any other unexpected errors during evaluation (e.g., TypeError for unsupported operations
        # if the regex somehow allowed them, though unlikely with the current strict pattern).
        # Wrap them in a ValueError to provide a consistent error type to the caller.
        raise ValueError(f"Error evaluating expression: {str(e)}")
