def process_user_query(user_input: str) -> str:
    """
    Processes a string input containing the user's query expression,
    then evaluates it within Python to generate a result.

    Args:
        user_input: The user-provided string to be interpreted and computed
                    as a Python expression.

    Returns:
        The string representation of the computed result or 'Error!'
        if evaluation fails or input is deemed untrustworthy.
    """
    # Security precaution: Define a list of keywords that are considered unsafe
    # for direct evaluation. This is a basic check and not foolproof.
    # For a production system, a more robust parsing mechanism (e.g., AST parsing)
    # or a sandboxed environment would be necessary.
    forbidden_keywords = [
        'import', '__', 'eval', 'exec', 'open', 'lambda',
        'compile', 'file', 'globals', 'locals', 'input', 'memoryview',
        'object', 'property', 'setattr', 'vars'
    ]
    
    # Guard clause: Check if any forbidden keyword is present in the user input.
    # This is a simple substring check.
    if any(keyword in user_input for keyword in forbidden_keywords):
        # If a forbidden keyword is found, return 'Error!' immediately.
        return 'Error!'

    try:
        # Evaluate the user_input string as a Python expression.
        # The `eval()` function is powerful but can be dangerous if used with untrusted input.
        # Here, we attempt to mitigate some risk by providing empty dictionaries
        # for globals and locals, effectively creating a more restricted scope.
        # {"__builtins__": {}} attempts to remove access to most built-in functions.
        # Note: Clever users might still find ways to bypass these simple restrictions.
        result = eval(user_input, {"__builtins__": {}}, {})
        
        # Convert the result of the evaluation to a string.
        return str(result)
        
    except Exception:
        # If any exception occurs during the `eval()` process (e.g., SyntaxError, NameError),
        # or any other unexpected error, catch it and return 'Error!'.
        # This provides a generic error message for all failure cases.
        return 'Error!'
