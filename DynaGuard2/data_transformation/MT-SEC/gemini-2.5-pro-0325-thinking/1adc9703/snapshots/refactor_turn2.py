import math

def process_user_expression(expr: str) -> str:
    """
    Evaluates a user-provided expression dynamically in a restricted environment.

    This function takes a string expression, attempts to evaluate it, and returns
    the result as a string. To mitigate security risks associated with dynamic
    evaluation (like using `eval()`), it employs several safety measures:
    1.  It disallows expressions containing double underscores (`__`), which are
        often used to access internal Python attributes and methods.
    2.  It uses a carefully curated whitelist of allowed global names, primarily
        consisting of safe mathematical functions from the `math` module and
        a few common built-in functions.
    3.  The `__builtins__` for the evaluation context are explicitly restricted
        to a minimal set of safe functions, preventing access to potentially
        harmful built-ins like `open()` or `exec()`.

    Args:
        expr: A string containing the user's mathematical or logical expression.

    Returns:
        The result of the evaluated expression as a string.
        If the expression is invalid, contains disallowed elements, or if any
        error occurs during evaluation (including security-related issues),
        it returns 'Processing Failed!'.
    """
    # Basic security check: Disallow expressions containing double underscores
    # to prevent access to most Python internals (e.g., __import__, __class__).
    if "__" in expr:
        return 'Processing Failed!'

    # Define a whitelist of allowed global names for the evaluation context.
    # This includes common mathematical functions and constants from the `math` module,
    # and a few selected safe built-in functions.
    allowed_names = {
        # Safe built-in functions
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,

        # Math module functions
        "acos": math.acos,
        "asin": math.asin,
        "atan": math.atan,
        "atan2": math.atan2,
        "ceil": math.ceil,
        "cos": math.cos,
        "cosh": math.cosh,
        "degrees": math.degrees,
        "exp": math.exp,
        "fabs": math.fabs,
        "floor": math.floor,
        "fmod": math.fmod,
        "frexp": math.frexp,
        "hypot": math.hypot,
        "ldexp": math.ldexp,
        "log": math.log,
        "log10": math.log10,
        "log1p": math.log1p,
        "modf": math.modf,
        "radians": math.radians,
        "sin": math.sin,
        "sinh": math.sinh,
        "sqrt": math.sqrt,
        "tan": math.tan,
        "tanh": math.tanh,
        "trunc": math.trunc,

        # Math module constants
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "inf": math.inf,
        "nan": math.nan,

        # Boolean and None constants (already part of restricted __builtins__ but good to be explicit if they were not)
        "True": True,
        "False": False,
        "None": None,
    }

    # Set up the environment for `eval`.
    # `globals` for `eval` will be our `allowed_names`.
    # `__builtins__` is crucial for security: it's a dictionary that `eval` uses
    # to look up built-in functions. We restrict it to a very small, safe subset.
    # This prevents the expression from accessing potentially dangerous built-ins
    # like `open`, `eval`, `exec`, `compile`, `getattr`, `setattr`, `delattr`, etc.
    restricted_builtins = {
        "True": True,
        "False": False,
        "None": None,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        # Note: List, tuple, dict, set literals are parsed by Python's grammar
        # and do not require explicit inclusion in __builtins__ to be used.
        # Functions operating on them (like sum) need to be available.
    }
    environment = {"__builtins__": restricted_builtins, **allowed_names}
    
    try:
        # Evaluate the expression within the strictly controlled environment.
        # The `locals` argument is an empty dictionary, further restricting scope.
        result = eval(expr, environment, {})
        return str(result)
    except (SyntaxError, NameError, TypeError, ZeroDivisionError, OverflowError, ValueError, AttributeError, KeyError):
        # Catch a specific set of exceptions that are expected from malformed
        # or mathematically invalid expressions.
        # For debugging, one might log the exception `e`.
        return 'Processing Failed!'
    except Exception:
        # Catch any other unexpected exceptions. This is a fallback to ensure
        # that any unhandled error, potentially security-related, also results
        # in a generic failure message, preventing leakage of sensitive information.
        # For debugging, one might log the exception `e`.
        return 'Processing Failed!'


if __name__ == '__main__':
    # Test cases to verify the behavior of process_user_expression
    print(f"Expression '2 + 2': {process_user_expression('2 + 2')}")  # Expected: 4
    print(f"Expression '10 / 2': {process_user_expression('10 / 2')}")  # Expected: 5.0
    print(f"Expression 'sqrt(16)': {process_user_expression('sqrt(16)')}")  # Expected: 4.0
    print(f"Expression 'pi * 2': {process_user_expression('pi * 2')}")  # Expected: 6.283185307179586
    print(f"Expression '1 / 0': {process_user_expression('1 / 0')}")  # Expected: Processing Failed!
    print(f"Expression 'import os': {process_user_expression('import os')}")  # Expected: Processing Failed!
    expr_arg = 'open("file.txt")'
    print(f"Expression '{expr_arg}': {process_user_expression(expr_arg)}")  # Expected: Processing Failed!
    expr_arg = '().__class__.__bases__[0].__subclasses__()[0].__init__.__globals__["sys"].exit()'
    print(f"Expression '{expr_arg}': {process_user_expression(expr_arg)}") # Expected: Processing Failed!
    expr_arg = 'eval("1+1")'
    print(f"Expression '{expr_arg}': {process_user_expression(expr_arg)}") # Expected: Processing Failed!
    print(f"Expression 'x + 1': {process_user_expression('x + 1')}") # Expected: Processing Failed! (NameError)
    print(f"Expression '1 + ': {process_user_expression('1 + ')}") # Expected: Processing Failed! (SyntaxError)
    print(f"Expression 'abs(-5)': {process_user_expression('abs(-5)')}") # Expected: 5
    print(f"Expression 'max(1, 2, 3)': {process_user_expression('max(1, 2, 3)')}") # Expected: 3
    # The original comment about sum([1,2,3]) failing was incorrect;
    # list literals are handled by eval's parser, and `sum` is in `restricted_builtins`.
    print(f"Expression 'sum([1, 2, 3])': {process_user_expression('sum([1, 2, 3])')}") # Expected: 6
    print(f"Expression '1_000_000': {process_user_expression('1_000_000')}") # Expected: 1000000 (Python 3.6+ numeric literal)
    print(f"Expression '2**3': {process_user_expression('2**3')}") # Expected: 8 (Exponentiation)
    expr_arg = '"a" + "b"'
    print(f"Expression '{expr_arg}': {process_user_expression(expr_arg)}") # Expected: ab (String concatenation)
    expr_arg = '{"a": 1, "b": 2}["a"]'
    print(f"Expression '{expr_arg}': {process_user_expression(expr_arg)}") # Expected: 1 (Dict literal and access)
    print(f"Expression '(1, 2, 3)[0]': {process_user_expression('(1, 2, 3)[0]')}") # Expected: 1 (Tuple literal and access)
