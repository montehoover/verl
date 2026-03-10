def calculate_expression(expr_input: str) -> str:
    """
    Evaluates a user-provided mathematical expression string.

    Args:
        expr_input: A string containing a basic mathematical expression.

    Returns:
        The result of the evaluation as a string, or 'Computation Error!'
        if the input is suspicious, unsafe, or causes an evaluation failure.
    """
    try:
        # Limit the length of the expression string to prevent abuse.
        # An arbitrary limit, e.g., 200 characters.
        if len(expr_input) > 200:
            return "Computation Error!"

        # Define a restricted environment for eval.
        # __builtins__ being an empty dictionary disallows calling any built-in functions.
        # Operators like +, -, *, / are handled by eval's parser directly.
        # Literals (numbers) are also handled.
        allowed_globals = {"__builtins__": {}}
        # No local variables from the surrounding scope are allowed.
        allowed_locals = {}

        result = eval(expr_input, allowed_globals, allowed_locals)

        # Ensure the result of the evaluation is a number (integer or float).
        # This prevents expressions that evaluate to strings, lists, etc.
        if not isinstance(result, (int, float)):
            return "Computation Error!"

        return str(result)

    except (SyntaxError, ZeroDivisionError, NameError, TypeError, OverflowError, MemoryError):
        # Catch common errors that can occur during evaluation of arithmetic expressions:
        # - SyntaxError: Malformed expression (e.g., "1 +* 2").
        # - ZeroDivisionError: Division by zero (e.g., "1 / 0").
        # - NameError: Undefined variable or function (e.g., "x + 1", "sqrt(4)").
        # - TypeError: Operation on an inappropriate type (e.g., if an expression somehow
        #              resulted in a non-numeric type that was then used in an operation).
        # - OverflowError: Result too large to be represented (e.g., "10**1000000").
        # - MemoryError: Expression tries to allocate too much memory.
        return "Computation Error!"
    except Exception:
        # Catch any other unexpected exceptions as a fallback.
        return "Computation Error!"
