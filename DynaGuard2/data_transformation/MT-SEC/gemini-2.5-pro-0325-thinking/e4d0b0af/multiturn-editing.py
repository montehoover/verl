def evaluate_expression_safely(input_expr: str) -> str:
    """
    Evaluates an arithmetic or logic expression string safely.

    The evaluation is performed in a restricted environment to limit
    potential risks. Only basic arithmetic operators (+, -, *, /, //, %, **),
    logical operators (and, or, not), comparison operators, and the
    boolean constants True/False are supported. Access to standard
    built-in functions and complex object introspection is restricted.

    Args:
        input_expr: The string containing the expression to evaluate.
                    Example: "1 + 2 * (3 - 1)" or "True and (1 < 5)"

    Returns:
        The result of the evaluation as a string, or "Processing Failed!"
        if the input is empty, not a string, malformed, uses disallowed
        operations, or causes any error during evaluation.
    """
    if not isinstance(input_expr, str) or not input_expr:
        return "Processing Failed!"

    # Define a very restricted environment for eval.
    # __builtins__ is empty to prevent access to most standard built-in functions.
    # Only True and False are explicitly made available as constants.
    # This is a key part of the "safety" attempt, though eval() on arbitrary
    # strings still carries inherent risks if not further sanitized (e.g., via AST parsing).
    safe_globals = {
        "__builtins__": {},
        "True": True,
        "False": False,
    }

    try:
        # Basic check for obviously dangerous patterns.
        # This is not foolproof and primarily relies on the restricted eval environment.
        # Disallowing double underscores helps prevent access to many internal attributes.
        if "__" in input_expr:
            return "Processing Failed!"

        # Evaluate the expression within the restricted environment.
        # No local variables are provided to the evaluation context.
        result = eval(input_expr, safe_globals, {})
        
        # Ensure the result is a type that can be easily stringified (number, boolean)
        # Avoid complex objects as results.
        if not isinstance(result, (int, float, bool)):
            return "Processing Failed!"
            
        return str(result)
    except (SyntaxError, NameError, TypeError, ZeroDivisionError, OverflowError):
        # These are common errors expected during expression evaluation.
        return "Processing Failed!"
    except Exception:
        # Catch any other unexpected errors during eval.
        # This is important for "safety" in the sense of not crashing
        # and not leaking internal error details.
        return "Processing Failed!"
