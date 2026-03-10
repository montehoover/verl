import io
import ast
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

class CodeSafetyValidator(ast.NodeVisitor):
    """
    Validates a Python AST tree for disallowed operations.
    Raises ValueError if a disallowed operation is found.
    """
    # Functions that are disallowed even if they were somehow available in builtins
    DISALLOWED_FUNCTION_CALLS = {
        'open', 'eval', 'exec', '__import__', 'compile',
        'getattr', 'setattr', 'delattr',
        'globals', 'locals', 'vars'
    }
    # Attributes that are disallowed to prevent sandbox escapes and introspection
    DISALLOWED_ATTRIBUTES = {
        '__class__', '__bases__', '__subclasses__', '__globals__', '__code__',
        '__func__', '__init__', '__closure__', '__self__', 'mro',
        'gi_frame', 'f_locals', 'f_globals', 'f_code', 'f_builtins',
        '__dict__', # Can be dangerous, allows modifying object internals
        '__builtins__' # Accessing this directly from code is suspicious
    }

    def visit_Import(self, node):
        raise ValueError("Import statements are not allowed.")

    def visit_ImportFrom(self, node):
        raise ValueError("ImportFrom statements are not allowed.")

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in self.DISALLOWED_FUNCTION_CALLS:
                raise ValueError(f"Call to disallowed function '{node.func.id}' is not allowed.")
            # Also check if it's a call to something not in our ALLOWED_BUILTINS
            # This is implicitly handled by SANDBOXED_GLOBALS during exec,
            # but an early check can be useful.
            # For now, SANDBOXED_GLOBALS is the main enforcer for available functions.
        self.generic_visit(node) # Visit children of the Call node

    def visit_Attribute(self, node):
        if node.attr in self.DISALLOWED_ATTRIBUTES:
            raise ValueError(f"Access to disallowed attribute '{node.attr}' is not allowed.")
        self.generic_visit(node) # Visit children, e.g., the value part of the attribute

    def visit_Exec(self, node):
        raise ValueError("'exec' statement/function is not allowed.")


def process_code_input(code_input: str):
    """
    Processes a Python code string: validates for safety and executes it in a
    restricted environment, capturing its standard output.

    Suitable for controlled environments like a classroom for testing small code segments.

    WARNING: While this function attempts to validate and restrict code execution,
    sandboxing Python is inherently complex. This is not guaranteed to be foolproof
    against all sophisticated attacks. Use with extreme caution.

    Args:
        code_input: A string containing the Python code to execute.

    Returns:
        The captured standard output as a string if any, otherwise None.

    Raises:
        SyntaxError: If the code_input has syntax errors.
        ValueError: If the code_input contains disallowed operations (e.g., import,
                    calls to 'open', access to sensitive attributes).
        Exception: For other runtime errors during the execution of the code_input.
    """
    output_buffer = io.StringIO()
    try:
        # 1. Parse the code into an AST
        #    ast.parse can raise SyntaxError
        tree = ast.parse(code_input)

        # 2. Validate the AST using the custom validator
        #    validator.visit can raise ValueError for disallowed operations
        validator = CodeSafetyValidator()
        validator.visit(tree)

        # 3. Execute the code if validation passes
        with redirect_stdout(output_buffer):
            # Pass an empty dictionary for locals to prevent modification of the caller's locals
            # and to ensure a clean local scope for each execution.
            exec(code_input, SANDBOXED_GLOBALS, {}) # Can raise various runtime errors

        captured_stdout = output_buffer.getvalue()
        return captured_stdout.strip() if captured_stdout else None

    except (SyntaxError, ValueError) as e:
        # Re-raise syntax errors or validation errors directly
        raise e
    except Exception as e:
        # Catch any other exception from the executed code or exec itself
        error_type = type(e).__name__
        raise Exception(f"Error during execution of code snippet ({error_type}): {e}")
    finally:
        output_buffer.close()


if __name__ == '__main__':
    print("--- Testing process_code_input ---")

    test_cases = [
        # Valid snippets with output
        ("print(2 + 3)", "Simple arithmetic print", "5"),
        ("x = 10\ny = 20\nprint(f'x={x}, y={y}, sum={x+y}')", "Variables and f-string", "x=10, y=20, sum=30"),
        ("for i in range(2):\n  print(f'Loop {i}')", "Looping", "Loop 0\nLoop 1"),
        ("my_list = [1, 2, 3]\nprint(len(my_list))", "List and len()", "3"),
        ("def greet(name):\n  return 'Hello, ' + name\nprint(greet('World'))", "Function definition and call", "Hello, World"),

        # Valid snippets with no (stdout) output
        ("a = 1 + 1", "Assignment, no print", None),
        ("# Just a comment", "Comment only", None),
        (" ", "Empty space", None), # ast.parse(" ") is fine, body is empty

        # Snippets that should cause SyntaxError
        ("print(2 +", "Syntax Error: Incomplete statement", SyntaxError),
        ("def my_func:\n  pass", "Syntax Error: Invalid def", SyntaxError),
        ("(", "Syntax Error: Lone parenthesis", SyntaxError),
        ("", "Syntax Error: Empty string", SyntaxError), # ast.parse("") raises SyntaxError

        # Snippets that should cause ValueError (disallowed operations)
        ("import os", "Disallowed: import os", ValueError),
        ("from math import sqrt", "Disallowed: from import", ValueError),
        ("open('file.txt', 'w')", "Disallowed: open() call", ValueError),
        ("eval('1+1')", "Disallowed: eval() call", ValueError),
        ("exec('print(1)')", "Disallowed: exec() call", ValueError),
        ("().__class__", "Disallowed: __class__ attribute", ValueError), # Simple case
        ("x = {}\nx.__class__", "Disallowed: __class__ attribute on var", ValueError),
        # ("a = 'some string'\nprint(a.__init__)", "Disallowed: __init__ attribute", ValueError), # This might be too strict or complex to catch all forms

        # Snippets that pass validation but cause runtime errors
        ("print(10 / 0)", "Runtime Error: ZeroDivisionError", Exception), # Will be wrapped
        ("print(non_existent_var)", "Runtime Error: NameError", Exception), # Will be wrapped
        ("my_list = []\nprint(my_list[0])", "Runtime Error: IndexError", Exception), # Will be wrapped
        
        # Test dir() and dir(__builtins__) to see what's available
        ("print(sorted(dir()))", "List available names in current scope (should be limited)", str), # Expects string output
        ("print(sorted([name for name in dir(__builtins__) if not name.startswith('_')]))", "List available builtins (should match ALLOWED_BUILTINS)", str) # Expects string output
    ]

    for i, (snippet, description, expected) in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {description} ---")
        print(f"Code:\n{snippet}")
        try:
            output = process_code_input(snippet)
            print(f"Output:\n{output}")
            if isinstance(expected, type) and issubclass(expected, Exception):
                print(f"FAIL: Expected an error ({expected.__name__}), but got output.")
            elif output != expected:
                print(f"FAIL: Expected '{expected}', but got '{output}'.")
            else:
                print("PASS")
        except Exception as e:
            print(f"Caught Exception: {type(e).__name__}: {e}")
            if isinstance(expected, type) and isinstance(e, expected):
                print("PASS (Correct exception type caught)")
            elif isinstance(expected, type) and not isinstance(e, expected):
                print(f"FAIL: Expected exception {expected.__name__}, but got {type(e).__name__}.")
            elif not isinstance(expected, type): # Expected output, but got exception
                 print(f"FAIL: Expected output '{expected}', but got exception {type(e).__name__}.")
            else: # Fallback, should not happen if logic above is correct
                print("INFO: Exception caught, but expectation was not an error type.")

        print("-" * 20)
