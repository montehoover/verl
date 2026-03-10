import logging

# Setup logger for this module.
# This allows the module to log output if run standalone or if the
# calling application hasn't configured logging.
# The basicConfig will only set up handlers if the root logger has no handlers.
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Helper functions

def _validate_input(input_expr: str) -> bool:
    """
    Validates the input expression.
    Currently checks if the input is a string.
    """
    if not isinstance(input_expr, str):
        return False
    # Future enhancements could include length checks or character set validation.
    return True

def _prepare_safe_environment() -> tuple[dict, dict]:
    """
    Prepares and returns the safe global and local environments for eval().
    The environment restricts built-ins and provides a whitelist of safe functions
    and constants. This acts as the "sanitization" step by controlling what
    the expression can access.
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

    # The 'globals' dictionary for eval must contain '__builtins__'.
    # By setting '__builtins__' to an empty dictionary, we effectively remove access
    # to all standard built-in functions and names by default,
    # preventing calls to potentially unsafe functions like open(), exec(), __import__(), etc.
    restricted_globals = {"__builtins__": {}}
    
    return allowed_names, restricted_globals

def _execute_computation(expression: str, globals_env: dict, locals_env: dict) -> any:
    """
    Dynamically computes the result of the expression string using eval().
    This function is intended to be called within a try-except block by the orchestrator,
    as eval() can raise various exceptions (SyntaxError, NameError, TypeError, etc.).
    """
    # The expression is evaluated with the restricted globals and whitelisted locals.
    return eval(expression, globals_env, locals_env)


# Main function (refactored)
def evaluate_expression_safely(input_expr: str) -> str:
    """
    Evaluates a user-input expression string safely by orchestrating
    validation, environment setup, and computation steps.
    Logs the expression, its evaluation outcome, or any errors.

    Args:
        input_expr: The string containing the expression to be evaluated.

    Returns:
        The string representation of the expression's result if successful,
        otherwise 'Processing Failed!' if the input is unsafe or results
        in an error during evaluation.
    """
    logger.info(f"Received expression for evaluation: '{input_expr}'")

    if not _validate_input(input_expr):
        logger.warning(f"Input validation failed for expression: '{input_expr}'. Outcome: 'Processing Failed!'")
        return 'Processing Failed!'

    allowed_names, restricted_globals = _prepare_safe_environment()
    
    try:
        # The _execute_computation function calls eval(), which can raise various exceptions, e.g.:
        # - SyntaxError: if the expression is not valid Python.
        # - NameError: if a variable or function name is not found in allowed_names.
        # - TypeError: for operations on incompatible types.
        # - ZeroDivisionError: for division by zero.
        # - OverflowError: for results too large to represent.
        # - MemoryError: if the expression consumes too much memory.
        # All such exceptions are caught by this general 'except Exception' block.
        
        result = _execute_computation(input_expr, restricted_globals, allowed_names)
        
        # Convert the result to a string, as per the function's specified return type.
        str_result = str(result)
        logger.info(f"Expression '{input_expr}' evaluated successfully. Result: '{str_result}'")
        return str_result
    except Exception as e:
        # If any error occurs during input validation, parsing, or evaluation,
        # or if the input attempts to use disallowed features (which would typically
        # lead to a NameError because the name isn't in 'allowed_names'),
        # return the specified failure message.
        logger.error(f"Error evaluating expression '{input_expr}'. Outcome: 'Processing Failed!'. Exception: {e}", exc_info=True)
        return 'Processing Failed!'
