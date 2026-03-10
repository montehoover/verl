def process_user_expression(expr: str):
    """
    Evaluates a user-provided arithmetic expression string.

    Args:
        expr: A string containing the arithmetic expression.

    Returns:
        The result of the evaluated expression, or "Processing Failed!"
        if the input is invalid or poses a security risk.
    """
    try:
        # Basic check for allowed characters to mitigate some security risks with eval().
        # For a production system, a more robust parsing solution (e.g., ast.literal_eval
        # for simple literals, or a dedicated expression parsing library) is recommended
        # over direct use of eval() with untrusted input.
        allowed_chars = set("0123456789+-*/(). ")
        if not all(char in allowed_chars for char in expr):
            return "Processing Failed!"

        # Ensure the expression is not empty or just whitespace.
        if not expr.strip():
            return "Processing Failed!"

        # Evaluate the expression.
        # Note: eval() can execute arbitrary code if not carefully restricted.
        # The character check above provides a basic level of safety for simple arithmetic.
        result = eval(expr)
        return result
    except (SyntaxError, NameError, TypeError, ZeroDivisionError, OverflowError, MemoryError):
        # Catching a broad range of errors that might occur during evaluation.
        return "Processing Failed!"
