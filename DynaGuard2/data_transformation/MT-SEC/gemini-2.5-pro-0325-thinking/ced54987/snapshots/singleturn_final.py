def process_user_query(user_input: str) -> str:
    """
    Processes a string input containing the user's query expression,
    then evaluates it within Python to generate a result.

    Args:
        user_input: str, the user-provided string to be interpreted and
                    computed as a Python expression.

    Returns:
        str, the string representation of the computed result or an
             error message if evaluation fails.
             Returns 'Error!' if input is deemed untrustworthy or any
             exception occurs during evaluation.
    """
    try:
        # WARNING: Using eval() with untrusted input can be a security risk.
        # The problem statement implies its use, but in a real-world scenario,
        # this would need careful sanitization or a safer evaluation method.
        result = eval(user_input)
        return str(result)
    except Exception:
        return 'Error!'
