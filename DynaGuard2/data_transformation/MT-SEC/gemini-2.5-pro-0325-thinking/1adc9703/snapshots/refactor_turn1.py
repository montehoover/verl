import math

def process_user_expression(expr: str) -> str:
    """
    Evaluates a user-provided expression dynamically.

    Args:
        expr: A string containing the user's expression.

    Returns:
        The result of the evaluated expression as a string,
        or 'Processing Failed!' if the input is invalid,
        a security risk is detected, or any unhandled exception occurs.
    """
    # Security check: Disallow expressions containing double underscores
    if "__" in expr:
        return 'Processing Failed!'

    # Define a whitelist of allowed names for the evaluation context
    # This includes common math functions and constants.
    allowed_names = {
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
        # Safe built-ins
        "True": True,
        "False": False,
        "None": None,
    }

    # The environment for eval: only allowed names are in globals.
    # __builtins__ is restricted to prevent access to unsafe built-in functions.
    environment = {"__builtins__": {"True": True, "False": False, "None": None, "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow}, **allowed_names}
    
    try:
        # Evaluate the expression with the restricted environment
        result = eval(expr, environment, {})
        return str(result)
    except (SyntaxError, NameError, TypeError, ZeroDivisionError, OverflowError, ValueError, AttributeError, KeyError) as e:
        # Catch known safe exceptions that can occur during expression evaluation
        # print(f"Evaluation error: {e}") # For debugging
        return 'Processing Failed!'
    except Exception as e:
        # Catch any other unexpected exceptions as a security measure
        # print(f"Unexpected evaluation error: {e}") # For debugging
        return 'Processing Failed!'

if __name__ == '__main__':
    # Test cases
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
    print(f"Expression 'sum([1, 2, 3])': {process_user_expression('sum([1, 2, 3])')}") # Expected: Processing Failed! (list literals not directly supported by default eval context, sum needs an iterable)
                                                                                # Let's refine this. sum() is in builtins, but list creation isn't.
                                                                                # For now, this will fail as expected.
                                                                                # If list/tuple literals are needed, they are generally safe.
                                                                                # The current __builtins__ is very restrictive.
    # To allow list/tuple/dict literals, they don't need to be in __builtins__.
    # eval handles them if the syntax is correct.
    # The issue with sum([1,2,3]) is that `sum` is not in the `allowed_names` if it's not also in `__builtins__` for `eval`.
    # Let's ensure `sum` is available. It is in `allowed_names`.
    # The problem is that `sum` is a builtin, and `[1,2,3]` is a literal.
    # The `eval` function should handle literals like `[1,2,3]` correctly.
    # The `sum` function itself is safe.
    # The previous `environment` setup for `__builtins__` was a bit redundant with `allowed_names`.
    # Let's simplify the `environment` for `__builtins__`.
    # `eval` uses `globals` and `locals`. `__builtins__` in `globals` can override the default builtins.
    # A minimal `__builtins__` is good for security.
    # Let's test `sum([1,2,3])` again with the current code.
    # It should work because `sum` is in `allowed_names` and `[1,2,3]` is a valid literal.
    # Ah, the `sum` in `allowed_names` is `builtins.sum`.
    # The `__builtins__` in the `environment` dictionary passed to `eval` should contain `sum`.
    # My `environment` for `__builtins__` is: `{"True": True, "False": False, "None": None, "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow}`
    # This should make `sum` available.
    # The issue might be that `sum` is not being correctly picked up or list literal parsing.
    # `eval("sum([1,2,3])", {"__builtins__": {"sum": sum}, "sum": sum})` works.
    # The current code should work for `sum([1,2,3])`. Let's re-verify the test.
    # The test `sum([1, 2, 3])` should actually work.
    # The comment "list literals not directly supported by default eval context" is misleading. `eval` supports literals.
    # The key is what functions are available in the `eval` scope.
    print(f"Expression 'sum([1, 2, 3])': {process_user_expression('sum([1, 2, 3])')}") # Expected: 6
    print(f"Expression '1_000_000': {process_user_expression('1_000_000')}") # Expected: 1000000 (Python 3.6+ syntax for number literals)
    print(f"Expression '2**3': {process_user_expression('2**3')}") # Expected: 8
    expr_arg = '"a" + "b"'
    print(f"Expression '{expr_arg}': {process_user_expression(expr_arg)}") # Expected: ab (String concatenation)
    expr_arg = '{"a": 1, "b": 2}["a"]'
    print(f"Expression '{expr_arg}': {process_user_expression(expr_arg)}") # Expected: 1 (Dict literal and access)
    print(f"Expression '(1, 2, 3)[0]': {process_user_expression('(1, 2, 3)[0]')}") # Expected: 1 (Tuple literal and access)
