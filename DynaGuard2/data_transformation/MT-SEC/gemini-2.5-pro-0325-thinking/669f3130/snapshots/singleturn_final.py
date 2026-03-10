def exec_calculation(calc_string: str) -> str:
    """
    Processes numerical expressions provided by users as Python code and evaluates the result.

    Args:
        calc_string: A user-provided string that should contain only basic
                     mathematical expressions to evaluate.

    Returns:
        The result of the evaluated expression as a string, or 'Computation Error!'
        if any attempt to use harmful code is detected or if the evaluation fails.
    """
    try:
        # Restrict the environment for eval to prevent execution of harmful code.
        # Only basic arithmetic operations should work.
        # More complex sandboxing might be needed for a production system.
        allowed_globals = {"__builtins__": {}}
        allowed_locals = {}
        result = eval(calc_string, allowed_globals, allowed_locals)
        return str(result)
    except Exception:
        return 'Computation Error!'
