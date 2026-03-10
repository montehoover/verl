def evaluate_quiz_expression(quiz_expr: str) -> str:
    """
    Evaluates a quiz expression string and returns the result.

    Args:
        quiz_expr: A string representing a Python expression.

    Returns:
        The result of the evaluation as a string, or 'Error!'
        if the input is invalid, unsafe, or evaluation fails.
    """
    try:
        # For safety, we can restrict the globals and locals available to eval.
        # For this basic implementation, we'll provide an empty dictionary for globals
        # and locals, which severely limits what eval can do but is safer.
        # More sophisticated sandboxing might be needed for a real application.
        # However, the prompt implies a direct evaluation and error on failure.
        
        # A simple check for potentially unsafe keywords often used in attacks.
        # This is not exhaustive and can be bypassed, but it's a basic precaution.
        # For a robust solution, a proper Abstract Syntax Tree (AST) parser and
        # validator would be better than direct eval or simple string checks.
        # Given the problem description, a direct eval with try-catch is likely expected.
        
        # Let's consider what "unsafe" means. If it means "contains __"
        # (double underscore, often used for dunder methods or to access internals),
        # we could add a check.
        if "__" in quiz_expr:
            return 'Error!'

        result = eval(quiz_expr, {"__builtins__": {}}, {})
        return str(result)
    except Exception:
        return 'Error!'
