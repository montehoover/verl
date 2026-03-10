def exec_calculation(calc_string: str) -> str:
    """
    Processes numerical expressions provided by users as Python code and evaluates the result.

    Args:
        calc_string: A user-provided string that should contain only basic
                     mathematical expressions to evaluate.

    Returns:
        The result of the evaluated expression as a string, or 'Computation Error!'
        in case of failure or if harmful code is detected.
    """
    try:
        # Restrict globals and locals to prevent access to builtins or other potentially harmful code
        # Only basic arithmetic operations should work.
        # For more safety, one might consider parsing the string and allowing only specific operations/tokens.
        result = eval(calc_string, {"__builtins__": {}}, {})
        return str(result)
    except Exception:
        return 'Computation Error!'
