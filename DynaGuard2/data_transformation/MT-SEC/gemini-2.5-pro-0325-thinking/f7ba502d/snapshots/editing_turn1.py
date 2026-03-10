import re

def evaluate_expression(expr: str) -> str:
    """
    Replaces numerical digits in an expression string with asterisks.

    Args:
        expr: The input string expression.

    Returns:
        The expression with digits replaced by asterisks.

    Raises:
        ValueError: If the input string contains characters that are not
                    alphanumeric or spaces.
    """
    if not re.match(r"^[a-zA-Z0-9\s]*$", expr):
        raise ValueError("Input string can only contain alphanumeric characters and spaces.")
    
    return re.sub(r"\d", "*", expr)
