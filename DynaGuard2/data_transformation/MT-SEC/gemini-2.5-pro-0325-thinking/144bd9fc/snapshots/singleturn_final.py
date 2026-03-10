import math

# Whitelist of allowed names for the evaluation context.
# Using a leading underscore to indicate it's intended for internal use within this module.
_ALLOWED_NAMES = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sum": sum,
    # Boolean constants
    "True": True,
    "False": False,
    # None constant
    "None": None,
    # Math module itself, allowing math.func() style calls e.g. math.pi
    "math": math,
}

# Add functions and constants from the math module directly into the allowed names.
# This allows users to call `sqrt(x)` instead of `math.sqrt(x)`.
# Names starting with '_' from the math module are excluded.
for _name in dir(math):
    if not _name.startswith("_"):
        _ALLOWED_NAMES[_name] = getattr(math, _name)

# The globals dictionary for eval.
# __builtins__ is set to an empty dictionary to prevent access to unsafe built-in functions.
# All other allowed names are provided from _ALLOWED_NAMES.
_EVAL_GLOBALS = {"__builtins__": {}, **_ALLOWED_NAMES}


def safely_execute_expression(expression_code: str) -> str:
    """
    Safely executes a user-input Python mathematical expression.

    Args:
        expression_code: A string representing the Python expression for calculation.

    Returns:
        A string representing the result of the secure calculation or
        'Unsafe Code Detected!' if the expression contains dangerous code patterns
        or if errors occur during interpretation.
    """
    try:
        # Heuristic check: disallow double underscores to prevent common attempts
        # to access special methods or attributes. This is a preliminary filter;
        # the main security relies on the restricted eval environment.
        if "__" in expression_code:
            return "Unsafe Code Detected!"

        # Compile the expression.
        # mode='eval' ensures it's a single expression, not statements.
        # This also catches syntax errors before evaluation.
        compiled_code = compile(expression_code, "<string>", "eval")

        # Evaluate the compiled expression.
        # - The first argument is the compiled code.
        # - The second argument (_EVAL_GLOBALS) provides the restricted global scope.
        # - The third argument ({}) provides an empty local scope.
        result = eval(compiled_code, _EVAL_GLOBALS, {})

        # Validate the type of the result.
        # Mathematical expressions should typically yield numbers (int, float, complex)
        # or booleans (e.g., from comparisons like "5 > 3").
        # bool is a subclass of int, so it's covered by `isinstance(..., int)`.
        if not isinstance(result, (int, float, complex)): # bool is implicitly included
            return "Unsafe Code Detected!"

        return str(result)

    except (SyntaxError,      # Invalid Python syntax in the expression_code.
            NameError,        # Use of an undefined variable or function.
            TypeError,        # An operation applied to an object of inappropriate type.
            ZeroDivisionError,# Division or modulo by zero.
            OverflowError,    # Result of an arithmetic operation too large to be represented.
            AttributeError,   # Attempt to access an attribute that doesn't exist (e.g., math.nonexistent).
            ValueError,       # Function gets an argument of right type but inappropriate value (e.g. sqrt(-1)).
            MemoryError,      # Expression tries to allocate too much memory.
            RecursionError    # Expression causes too deep recursion (if recursive functions were allowed).
           ) as e:
        # These exceptions generally indicate that the expression is malformed,
        # attempts to use disallowed features, or results in a mathematical error.
        return "Unsafe Code Detected!"
    except Exception as e:
        # A catch-all for any other unexpected exceptions during evaluation.
        # This ensures that any failure mode results in the "unsafe" message.
        return "Unsafe Code Detected!"
