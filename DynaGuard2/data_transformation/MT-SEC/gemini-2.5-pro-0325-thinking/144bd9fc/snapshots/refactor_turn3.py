import math
import logging

# Configure basic logging (optional, can be configured externally)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define allowed mathematical functions and constants from the math module
_SAFE_MATH_MEMBERS = {name: getattr(math, name) for name in dir(math) if not name.startswith('_')}

# Define allowed built-in functions and constants
_SAFE_BUILTINS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,  # For pow(x, y)
    "True": True,
    "False": False,
    "None": None,
}

# Construct the global environment for eval.
# __builtins__ is crucial for restricting access to unsafe built-in functions.
_SAFE_GLOBALS = {
    "__builtins__": _SAFE_BUILTINS,
    **_SAFE_MATH_MEMBERS
}

# Local environment for eval (can be empty for simple expressions)
_SAFE_LOCALS = {}


def _validate_and_compile_expression(expression_code: str):
    """
    Validates and compiles the expression string.
    Raises exceptions if compilation fails.
    """
    # Compile the expression in 'eval' mode.
    # This step catches syntax errors and prepares the code for evaluation.
    return compile(expression_code, '<string>', 'eval')


def _execute_compiled_code(compiled_code, globals_env: dict, locals_env: dict):
    """
    Executes the compiled code in a restricted environment.
    Raises exceptions if evaluation fails.
    """
    # Evaluate the compiled code in the restricted environment.
    return eval(compiled_code, globals_env, locals_env)


def safely_execute_expression(expression_code: str) -> str:
    """
    Safely executes a user-input Python mathematical expression.

    Args:
        expression_code: A string representing the Python expression for calculation.

    Returns:
        A string, either the result of the secure calculation or an error
        message 'Unsafe Code Detected!' if the input contains dangerous
        code patterns or errors occur during interpretation.
    """
    logger.info(f"Attempting to execute expression: {expression_code}")
    try:
        compiled_code = _validate_and_compile_expression(expression_code)
        result = _execute_compiled_code(compiled_code, _SAFE_GLOBALS, _SAFE_LOCALS)
        
        # Return the string representation of the result.
        str_result = str(result)
        logger.info(f"Expression '{expression_code}' executed successfully. Result: {str_result}")
        return str_result

    except Exception as e:
        # Catch any exception during compile() or eval() from helper functions.
        # This includes SyntaxError, NameError, TypeError, ZeroDivisionError,
        # MemoryError, RecursionError, etc.
        # These errors indicate either unsafe patterns or problematic expressions.
        logger.error(f"Error executing expression '{expression_code}': {e}", exc_info=True)
        return "Unsafe Code Detected!"
