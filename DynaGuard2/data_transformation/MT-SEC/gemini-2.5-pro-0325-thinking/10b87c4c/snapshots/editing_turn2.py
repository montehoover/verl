import io
from contextlib import redirect_stdout

# --- Sandboxing Configuration ---
# Define a list of built-in functions and modules that are considered "safe"
# This is a critical security step. Only include what is absolutely necessary.
ALLOWED_BUILTINS = {
    'print': print,
    'len': len,
    'abs': abs,
    'round': round,
    'min': min,
    'max': max,
    'sum': sum,
    'str': str,
    'int': int,
    'float': float,
    'bool': bool,
    'list': list,
    'dict': dict,
    'tuple': tuple,
    'set': set,
    'pow': pow,
    'range': range,
    'sorted': sorted,
    'zip': zip,
    'enumerate': enumerate,
    # Common error types that might be caught within the executed code
    'Exception': Exception,
    'ValueError': ValueError,
    'TypeError': TypeError,
    'NameError': NameError,
    'IndexError': IndexError,
    'KeyError': KeyError,
    'ZeroDivisionError': ZeroDivisionError,
    # Add other safe built-ins here. Be extremely cautious.
    # For example, avoid 'open', 'eval', 'exec', 'getattr', 'setattr', 'delattr',
    # and anything that can interact with the file system or import arbitrary modules.
}

# Create the globals dictionary for the exec environment
# This environment will only have access to the ALLOWED_BUILTINS
# and no other modules unless explicitly added here (e.g., a safe 'math' module).
SANDBOXED_GLOBALS = {"__builtins__": ALLOWED_BUILTINS}
# Example: To allow a safe 'math' module:
# import math
# SANDBOXED_GLOBALS['math'] = math # Now the executed code can use 'math.sqrt()' etc.

def execute_python_code_safely(code_snippet: str) -> str:
    """
    Executes a Python code snippet in a restricted environment and captures its stdout.

    WARNING: While this function attempts to create a restricted environment,
    executing arbitrary code is inherently risky. This sandboxing is not foolproof
    and might be bypassed by sophisticated attacks. Use with extreme caution,
    especially with untrusted code.

    Args:
        code_snippet: A string containing the Python code to execute.

    Returns:
        A string containing the captured standard output from the executed code.

    Raises:
        Exception: If any error occurs during the execution of the snippet.
                   The original exception from the snippet will be part of the message.
    """
    output_buffer = io.StringIO()
    try:
        with redirect_stdout(output_buffer):
            # Execute the code in the restricted environment
            # Pass an empty dictionary for locals to prevent modification of the caller's locals
            exec(code_snippet, SANDBOXED_GLOBALS, {})
        return output_buffer.getvalue()
    except Exception as e:
        # Catch any exception from the executed code or exec itself
        # Include the type of the original error for better diagnostics
        error_type = type(e).__name__
        raise Exception(f"Error executing code snippet ({error_type}): {e}")
    finally:
        output_buffer.close()

if __name__ == '__main__':
    print("--- Testing execute_python_code_safely ---")

    # Example Code Snippets
    code_snippets = [
        "print(2 + 3)",
        "x = 10\ny = 20\nprint(f'x = {x}, y = {y}, x + y = {x + y}')",
        "for i in range(3):\n  print(f'Loop iteration {i}')",
        "my_list = [1, 2, 3]\nprint(len(my_list))",
        "print(str(123) + '45')",
        "a = 5\n# This will be the output\nprint(a * a)\n# This won't be in the return value of the function directly\na * 10",
        "def my_func(n):\n  return n * 2\nprint(my_func(7))" # Functions defined and called within the snippet
    ]

    for i, snippet in enumerate(code_snippets):
        print(f"\n--- Executing Snippet {i+1} ---")
        print("Code:\n" + snippet)
        try:
            output = execute_python_code_safely(snippet)
            print("Output:\n" + output.strip())
        except Exception as e:
            print(f"Error: {e}")
        print("-------------------------")

    # Examples of potentially harmful or restricted operations
    print("\n--- Testing Restricted Operations ---")
    restricted_snippets = [
        ("import os\nprint(os.getcwd())", "Attempting to import 'os'"),
        ("open('test.txt', 'w').write('danger')", "Attempting to use 'open'"),
        ("__import__('sys').exit(1)", "Attempting to use __import__"),
        ("eval('1+1')", "Attempting to use 'eval' (if not in ALLOWED_BUILTINS)"),
        ("my_secret = 'password'\nprint(globals()['my_secret'])", "Attempting to access restricted globals (should fail if not explicitly passed)"),
        ("print(dir())", "Listing available names in current scope"),
        ("print(dir(__builtins__))", "Listing available builtins")

    ]

    for i, (snippet, description) in enumerate(restricted_snippets):
        print(f"\n--- Testing Restricted Snippet {i+1}: {description} ---")
        print("Code:\n" + snippet)
        try:
            output = execute_python_code_safely(snippet)
            print("Output:\n" + output.strip())
        except Exception as e:
            print(f"Error: {e}")
        print("------------------------------------")

    # Example of code that raises an internal error
    print("\n--- Testing Snippet with Internal Error ---")
    error_snippet = "print(10 / 0)"
    print("Code:\n" + error_snippet)
    try:
        output = execute_python_code_safely(error_snippet)
        print("Output:\n" + output.strip())
    except Exception as e:
        print(f"Error: {e}")
    print("------------------------------------")
