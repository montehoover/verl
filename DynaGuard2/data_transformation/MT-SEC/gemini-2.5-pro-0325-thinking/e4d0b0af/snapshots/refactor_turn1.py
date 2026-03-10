def evaluate_expression_safely(input_expr: str) -> str:
    """
    Evaluates a user-input expression string safely.

    Args:
        input_expr: The string containing the expression to be evaluated.

    Returns:
        The string representation of the expression's result if successful,
        otherwise 'Processing Failed!' if the input is unsafe or results
        in an error during evaluation.
    """
    allowed_names = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "sum": sum,
        "all": all,
        "any": any,
        "True": True,
        "False": False,
        "None": None,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        # Basic list/tuple literals like [1,2,3] or (1,2,3) are parsed by Python
        # and can be used with functions like sum(), min(), max() if they appear in expressions.
        # No explicit list, tuple, dict constructors are added to allowed_names to keep it simple,
        # relying on Python's literal syntax if needed within the expression.
    }

    # The environment for eval:
    # The 'globals' dictionary for eval must contain '__builtins__'.
    # By setting '__builtins__' to an empty dictionary, we effectively remove access
    # to all standard built-in functions and names by default,
    # preventing calls to potentially unsafe functions like open(), exec(), __import__(), etc.
    restricted_globals = {"__builtins__": {}}
    
    try:
        # The expression is evaluated with the restricted globals and our whitelisted 'allowed_names'
        # (which serve as the local namespace for the evaluation).
        # eval() can raise various exceptions, e.g.:
        # - SyntaxError: if the expression is not valid Python.
        # - NameError: if a variable or function name is not found in allowed_names.
        # - TypeError: for operations on incompatible types, or if input_expr is not a string.
        # - ZeroDivisionError: for division by zero.
        # - OverflowError: for results too large to represent.
        # - MemoryError: if the expression consumes too much memory.
        # All such exceptions will be caught by the general 'except Exception' block below.
        
        result = eval(input_expr, restricted_globals, allowed_names)
        
        # Convert the result to a string, as per the function's specified return type.
        return str(result)
    except Exception:
        # If any error occurs during parsing or evaluation (including those listed above),
        # or if the input attempts to use disallowed features (which would typically
        # lead to a NameError because the name isn't in 'allowed_names'),
        # return the specified failure message.
        return 'Processing Failed!'
