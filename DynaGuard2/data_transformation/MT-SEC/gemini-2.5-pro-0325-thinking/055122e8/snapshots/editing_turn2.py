import math

def execute_operation(operation: str) -> float:
    """
    Executes a Python arithmetic expression string.

    Args:
        operation: A string representing a Python arithmetic expression,
                   e.g., '5 + 3', '3 * (4 + 5)', 'abs(-5.5)', 'math.sqrt(4)'.
                   Supports standard Python arithmetic operators, precedence,
                   and a limited set of safe functions from builtins and math module.

    Returns:
        The result of the expression as a float.

    Raises:
        SyntaxError: If the expression string is malformed.
        ZeroDivisionError: If the expression involves division by zero.
        NameError: If the expression uses undefined variables or unallowed functions.
        TypeError: If operations are attempted on incompatible types.
        ValueError: If the result of an expression cannot be converted to float,
                    or for other evaluation errors not covered by more specific exceptions.
    """
    # Define a dictionary of allowed names for eval's globals
    # This includes safe built-ins and functions from the math module
    allowed_names = {
        "abs": abs,
        "min": min,
        "max": max,
        "round": round,
        "sum": sum,
        "float": float,
        "int": int,
        "str": str, # Allow str for completeness, though direct use might be rare in arithmetic
        "math": math, # Exposes the math module (e.g., math.sqrt, math.pi)
        # For more fine-grained control, specific math functions can be added:
        # "pi": math.pi, "e": math.e, "sqrt": math.sqrt,
    }

    # Using eval to parse and compute the expression.
    # eval is powerful but can be risky if the input string is untrusted.
    # Here, we restrict its environment by providing a custom 'globals' dictionary.
    # The '__builtins__' key is set to an empty dict to prevent access to
    # all built-in functions, only allowing those explicitly in 'allowed_names'.
    try:
        # The locals dictionary is also empty.
        result = eval(operation, {"__builtins__": {}, **allowed_names}, {})
        # Ensure the final result is a float, as per function signature.
        return float(result)
    except (SyntaxError, ZeroDivisionError, NameError, TypeError) as e:
        # Re-raise known, expected exceptions from eval or float() conversion.
        raise e
    except Exception as e:
        # Wrap other, unexpected exceptions from eval or float() conversion in a ValueError.
        # This can help signal that the expression caused an issue not
        # typically expected from "safe" arithmetic or conversion.
        raise ValueError(f"Error evaluating expression: '{operation}'. Reason: {type(e).__name__} - {e}") from e

if __name__ == '__main__':
    # Example usage and basic tests
    test_cases = {
        # Basic arithmetic
        "5 + 3": 8.0,
        "10 - 2.5": 7.5,
        "4 * 2": 8.0,
        "10 / 2": 5.0,
        "10 / 4": 2.5,
        "2 ** 3": 8.0,
        "10 % 3": 1.0,
        "(2 + 3) * 5.0": 25.0,
        # Precedence
        "3 * (4 + 5)": 27.0,
        "(10 + 2) / 3": 4.0,
        # Allowed functions
        "abs(-5.5)": 5.5,
        "sum([1, 2, 3, 4])": 10.0,
        "min(1, 2, -1)": -1.0,
        "max(10, 20, 15)": 20.0,
        "round(3.14159, 2)": 3.14,
        # Math module usage
        "math.sqrt(9)": 3.0,
        "math.pi": math.pi,
        # Error cases
        "7 / 0": "ZeroDivisionError",
        "1 / 0.0": "ZeroDivisionError",
        "10 +": "SyntaxError",
        "10 +* 5": "SyntaxError",
        "a + 3": "NameError",
        "3 + b": "NameError",
        "sqrt(4)": "NameError", # Not directly available, use math.sqrt
        "__import__('os')": "NameError", # __import__ not in allowed_names
        "open('file.txt')": "NameError", # open not in allowed_names
        "eval('1+1')": "NameError",      # eval itself is not exposed
        # TypeErrors from operations or float conversion
        "'a' + 1": "TypeError", # eval itself raises: can only concatenate str (not "int") to str
        "'hello' * 2": "ValueError", # float('hellohello') raises ValueError, caught by general Exception
        "'hello' + 'world'": "ValueError", # float('helloworld') raises ValueError
        "math.sqrt(-1)": "ValueError", # math.sqrt(-1) raises ValueError (domain error), caught by general Exception
    }

    for op_str, expected in test_cases.items():
        print(f"Executing: '{op_str}'")
        try:
            result = execute_operation(op_str)
            if isinstance(expected, str): # Expected an error
                print(f"  ERROR: Expected {expected}, but got {result}")
            elif abs(result - expected) < 1e-9:
                print(f"  Success: {result}")
            else:
                print(f"  ERROR: Expected {expected}, got {result}")
        except (SyntaxError, ZeroDivisionError, NameError, TypeError, ValueError) as e:
            if isinstance(expected, str) and expected == type(e).__name__:
                print(f"  Success: Caught expected {type(e).__name__}: {e}")
            else:
                print(f"  ERROR: Unexpected {type(e).__name__}: {e}. Expected: {expected}")
        except Exception as e:
            print(f"  ERROR: An unexpected error occurred: {type(e).__name__}: {e}")
        print("-" * 20)
