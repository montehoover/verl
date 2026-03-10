import re

def run_user_code(python_code: str):
    """
    Executes a user-supplied Python script string.
    The script should only include basic arithmetic and string operations,
    and assign its final result to a variable named 'output'.
    Returns the value of 'output' or 'Execution Blocked!' if the script 
    is invalid, unsafe, or an error occurs.
    """
    # Allow letters, numbers, basic arithmetic operators, strings (single/double quotes),
    # parentheses, dot (for floats), e/E (for scientific notation), underscore,
    # equals sign (for assignment), and whitespace (including newlines).
    # This regex is a first-pass filter. Security primarily relies on the restricted exec() environment.
    allowed_chars_pattern = r"^[a-zA-Z0-9\s\+\-\*\/\(\)\.eE\'\"_=\n]*$"
    if not re.fullmatch(allowed_chars_pattern, python_code):
        return 'Execution Blocked!'

    # Prevent empty strings or strings with only whitespace.
    if not python_code.strip():
        return 'Execution Blocked!'

    local_scope = {}
    # Global scope for exec, with __builtins__ severely restricted.
    # This prevents access to most built-in functions and modules.
    restricted_globals = {"__builtins__": {}}

    try:
        # Execute the user's Python code.
        # Assignments and computations will occur in local_scope.
        exec(python_code, restricted_globals, local_scope)

        # Check if the script assigned a value to the 'output' variable.
        if 'output' in local_scope:
            result = local_scope['output']
            # Ensure the result is of an expected type (number or string).
            if isinstance(result, (int, float, str)):
                return result
            else:
                # Result is of an unexpected type.
                return 'Execution Blocked!'
        else:
            # The script did not assign to 'output' as per convention.
            return 'Execution Blocked!'

    except (SyntaxError, ZeroDivisionError, TypeError, NameError, OverflowError):
        # Catch common errors from arithmetic/string operations or undefined variables.
        return 'Execution Blocked!'
    except Exception:
        # Catch any other unexpected errors during execution.
        return 'Execution Blocked!'
