import re

def safe_eval_expression(expr: str):
    """Safely evaluate a string mathematical expression.

    This function takes a string representing a mathematical expression,
    validates it to ensure it only contains allowed characters (numbers,
    basic arithmetic operators, parentheses, and whitespace), and then
    evaluates it.

    Args:
        expr: The string containing the mathematical expression.
              Example: "1 + 2 * (3 - 1)"

    Returns:
        The numerical result of evaluating the mathematical expression.
        The type of the result will typically be int or float.

    Raises:
        ValueError: If the input expression string contains invalid
                    characters (e.g., letters, disallowed symbols) or
                    if the expression is syntactically incorrect (e.g.,
                    unbalanced parentheses, division by zero).
    """
    # Regex to validate the expression:
    # ^ : asserts position at start of the string.
    # [0-9\s\+\-\*\/\(\)\.] : matches any digit, whitespace, plus, minus,
    #                         asterisk, slash, parenthesis, or dot.
    # * : matches the previous token between zero and unlimited times.
    # $ : asserts position at the end of the string.
    # This ensures the entire string consists only of allowed characters.
    allowed_pattern = r"^[0-9\s\+\-\*\/\(\)\.]*$"
    if not re.match(allowed_pattern, expr):
        raise ValueError(
            "Invalid characters in expression. Only numbers, operators (+, -, *, /), "
            "parentheses, and whitespace are allowed."
        )

    try:
        # Evaluate the sanitized expression.
        # Note: While re.match provides some safety, using eval() directly
        # can still be risky if the sanitization is not perfect or if
        # the environment allows overriding built-ins. For highly secure
        # applications, consider using ast.literal_eval or a dedicated
        # expression parsing library.
        calculated_result = eval(expr)
        return calculated_result
    except (SyntaxError, ZeroDivisionError, TypeError, NameError) as e:
        # Catch common errors during evaluation and raise a ValueError.
        raise ValueError(f"Incorrect expression: {e}")
