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
    try:
        # For safety, it's better to restrict the environment for eval.
        # However, per prompt, a basic implementation is provided.
        # A more secure version would involve parsing the expression or
        # providing a very limited set of globals/locals to eval().
        # For now, any exception during eval will lead to 'Error!'.
        
        # A simple check for potentially unsafe keywords.
        # This is not exhaustive and can be bypassed.
        # A proper solution would involve an Abstract Syntax Tree (AST) parser.
        forbidden_keywords = ['import', '__', 'eval', 'exec', 'open', 'lambda']
        if any(keyword in user_input for keyword in forbidden_keywords):
            return 'Error!'

        result = eval(user_input, {"__builtins__": {}}, {}) # Restrict builtins for some safety
        return str(result)
    except Exception:
        return 'Error!'
