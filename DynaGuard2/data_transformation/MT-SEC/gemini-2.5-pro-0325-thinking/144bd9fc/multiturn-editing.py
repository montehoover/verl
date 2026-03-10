import re
import math
from typing import Any, Union

# Whitelist of functions and constants considered safe.
# Keys are the names available in the expression, values are the actual Python objects.
SAFE_GLOBALS = {
    "__builtins__": {},  # Start with no builtins for maximum security
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    # Math constants
    "pi": math.pi,
    "e": math.e,
    # Safe math functions
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "radians": math.radians,
    "degrees": math.degrees,
    "factorial": math.factorial,
}

def safely_execute_expression(expression_code: str) -> Union[Any, str]:
    """
    Safely executes a user-supplied Python expression string.

    Allows basic arithmetic and a whitelist of safe functions/constants.
    Aims to prevent execution of arbitrary, unsafe code.

    Args:
        expression_code: The Python expression as a string.

    Returns:
        The result of the evaluation,
        or an error message string if the expression is unsafe or invalid.
    """
    # Regex to allow:
    # - numbers (integers, floats, scientific notation like 1e-5)
    # - arithmetic operators (+, -, *, /)
    # - parentheses for grouping
    # - commas for function arguments
    # - dots for float numbers or attribute access on safe objects (none exposed here)
    # - alphanumeric characters and underscores for function/constant names from SAFE_GLOBALS.
    # This pattern is crucial for security. It must be restrictive.
    # It explicitly disallows characters like ';', '[', ']', '{', '}', ':', '"', "'", etc.,
    # which could be used for more complex statements or string literals that might hide code.
    # Note: `**` for power is not directly in this regex; use `pow(x,y)`.
    safe_pattern = r"^[a-zA-Z0-9\s\.\+\-\*\/\(\)\_\,eE]+$" # Added eE for scientific notation
    
    # Check for empty or whitespace-only strings
    if not expression_code.strip():
        return "Error: Expression cannot be empty."

    if not re.fullmatch(safe_pattern, expression_code):
        return "Error: Expression contains disallowed characters."

    try:
        # Execute the expression in a restricted environment.
        # `SAFE_GLOBALS` provides the allowed functions/constants.
        # An empty dictionary `{}` is passed for `locals`.
        result = eval(expression_code, SAFE_GLOBALS, {})
        return result
    except ZeroDivisionError:
        return "Error: Division by zero."
    except SyntaxError as e:
        return f"Error: Invalid syntax in expression - {e}."
    except NameError as e:
        # Extract the name from the NameError message if possible
        name_match = re.search(r"name '([^']*)' is not defined", str(e))
        name = name_match.group(1) if name_match else "unknown"
        return f"Error: Name '{name}' is not defined or not allowed."
    except TypeError as e:
        return f"Error: Type mismatch or operation not supported on given types - {e}."
    except OverflowError as e:
        return f"Error: Numerical result out of range - {e}."
    except Exception as e:
        # Catch-all for other potential errors during eval.
        return f"Error: An unexpected error occurred during execution: {type(e).__name__} - {e}."

if __name__ == '__main__':
    print("--- Testing safely_execute_expression ---")

    expressions_to_test = [
        "2 + 2",
        "10 - 4.5",
        "3 * 7",
        "20 / 5",
        "(2 + 3) * 4 / 2 - 1",
        "abs(-10)",
        "round(pi, 4)",
        "pow(2, 3)",
        "sqrt(16)",
        "log10(100)",
        "sin(radians(90))", # Should be close to 1
        "factorial(5)",
        "1e-5 * 100000", # Scientific notation
        "min(1, 2, -3) + max(4.5, 5, 6.1)",
        "pi", # Constant
        "e",  # Constant

        # Error cases
        "",                                 # Empty string
        "   ",                              # Whitespace only string
        "10 / 0",                           # Division by zero
        "5 + ",                             # Syntax error
        "1 + unknown_var",                  # Name error
        "import os",                        # Disallowed (SyntaxError for 'import' in eval)
        "eval('1+1')",                      # Disallowed (regex: quotes, then NameError: eval)
        "open('file.txt')",                 # Disallowed (regex: quotes, then NameError: open)
        "__import__('os').system('ls')",    # Disallowed (regex: quotes, dunders)
        "object()",                         # NameError: object
        "1; 2",                             # Disallowed (regex: semicolon)
        "a = 1",                            # Disallowed (SyntaxError for assignment in eval)
        "[1, 2, 3]",                        # Disallowed (regex: brackets)
        "{'a': 1}",                         # Disallowed (regex: braces, quotes)
        "lambda x: x + 1",                  # Disallowed (regex: colon)
        "sqrt(-1)",                         # Math domain error (ValueError, caught by generic Exception)
        "factorial(1000)",                  # OverflowError for large factorials
        "10**2",                            # Disallowed by regex (`**` not explicitly allowed)
        "def f(): return 1"                 # Disallowed (SyntaxError for 'def' in eval)
    ]

    for expr in expressions_to_test:
        result = safely_execute_expression(expr)
        print(f"Expression: '{expr if expr else '<empty>'}' -> Result: {result}")

    # Specific test for disallowed characters not in the loop
    expr_disallowed_chars = '1 + 1; print("hello")'
    print(f"Expression: '{expr_disallowed_chars}' -> Result: {safely_execute_expression(expr_disallowed_chars)}")
    # Specific test for dunder name usage in expression (should be caught by NameError or regex)
    expr_dunder_usage = '__builtins__["eval"]("1+1")'
    print(f"Expression: '{expr_dunder_usage}' -> Result: {safely_execute_expression(expr_dunder_usage)}")
