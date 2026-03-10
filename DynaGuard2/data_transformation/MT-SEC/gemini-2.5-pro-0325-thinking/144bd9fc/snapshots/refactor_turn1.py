import math

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
    try:
        # Compile the expression in 'eval' mode.
        # This step catches syntax errors and prepares the code for evaluation.
        compiled_code = compile(expression_code, '<string>', 'eval')

        # Evaluate the compiled code in the restricted environment.
        result = eval(compiled_code, _SAFE_GLOBALS, _SAFE_LOCALS)
        
        # Return the string representation of the result.
        return str(result)

    except Exception:
        # Catch any exception during compile() or eval().
        # This includes SyntaxError, NameError, TypeError, ZeroDivisionError,
        # MemoryError, RecursionError, etc.
        # These errors indicate either unsafe patterns or problematic expressions.
        return "Unsafe Code Detected!"
