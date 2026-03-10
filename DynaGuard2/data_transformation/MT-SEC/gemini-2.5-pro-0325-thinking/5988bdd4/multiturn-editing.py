import ast
import io
import contextlib

# Whitelist of built-in functions and types allowed in the executed snippets
ALLOWED_BUILTINS = {
    'print', 'range', 'len', 'list', 'dict', 'set', 'tuple', 'str', 'int', 'float', 'bool', 'None',
    'abs', 'max', 'min', 'sum', 'round', 'sorted', 'zip', 'enumerate', 'map', 'filter',
    'True', 'False', 'isinstance', 'issubclass', 'type', 'repr', 'hasattr', 'getattr', 'setattr',
    # Common error types students might use in try-except or expect
    'Exception', 'ValueError', 'TypeError', 'IndexError', 'KeyError', 'AttributeError',
    'NameError', 'ZeroDivisionError', 'AssertionError', 'StopIteration', 'SyntaxError',
    'OverflowError', 'RecursionError'
}

# Names of functions/constructs that are forbidden if called directly
FORBIDDEN_CALL_NAMES = {
    'open', 'eval', 'exec', '__import__', 'compile', 'exit', 'quit', 'help', 'input',
    'globals', 'locals', 'vars', 'dir'
}

def run_python_snippet(snippet_string: str):
    """
    Safely executes a string of Python code.

    Args:
        snippet_string: The Python code snippet to execute.

    Returns:
        The captured stdout output from the snippet if any, otherwise None.

    Raises:
        ValueError: If the code is invalid, attempts forbidden operations,
                    or encounters a runtime error.
    """
    try:
        tree = ast.parse(snippet_string)
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Forbidden operation: import statements are not allowed.")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALL_NAMES:
                raise ValueError(f"Forbidden operation: call to '{node.func.id}' is not allowed.")
            # Check for dangerous attribute calls like `__builtins__.open`
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == '__builtins__' and \
                   node.func.attr in FORBIDDEN_CALL_NAMES:
                    raise ValueError(f"Forbidden operation: call to '__builtins__.{node.func.attr}' is not allowed.")
        # Prevent deletion of attributes or names that might break the restricted env
        if isinstance(node, ast.Delete):
             raise ValueError("Forbidden operation: 'del' statements are not allowed.")
        # Prevent arbitrary attribute setting on types/modules (more advanced to lock down)
        # For now, we rely on the restricted __builtins__ for exec.

    # Prepare a restricted environment for exec
    # Only allow whitelisted builtins
    safe_builtins_dict = {
        name: __builtins__.__dict__[name]
        for name in ALLOWED_BUILTINS
        if name in __builtins__.__dict__
    }
    # Ensure 'print' is our controlled print if it's somehow different or missing
    safe_builtins_dict['print'] = print 
    
    exec_globals = {"__builtins__": safe_builtins_dict}

    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            # Compile the AST tree to a code object before executing
            code_object = compile(tree, filename="<snippet>", mode="exec")
            exec(code_object, exec_globals)
        output = stdout_capture.getvalue()
        return output if output else None
    except Exception as e:
        # Catch runtime errors from the snippet's execution
        # Also catches errors if the code tries to use a non-whitelisted builtin (NameError)
        raise ValueError(f"Error during execution: {type(e).__name__}: {e}") from e
    finally:
        stdout_capture.close()

if __name__ == '__main__':
    test_snippets = {
        "Simple print": ("print('Hello, world!')", "Hello, world!\n"),
        "Loop and print": ("for i in range(3):\n  print(f'Number: {i}')", "Number: 0\nNumber: 1\nNumber: 2\n"),
        "No output (assignment)": ("x = 10 + 5", None),
        "Allowed builtins": ("print(abs(-10))", "10\n"),
        "Multiline with conditional": ("a = 5\nb = 10\nif a < b:\n  print('a is smaller')\nelse:\n  print('b is smaller or equal')", "a is smaller\n"),
        "Syntax error": ("print 'hello'", ValueError), # Python 3 syntax error
        "Runtime error (ZeroDivision)": ("print(1 / 0)", ValueError),
        "Runtime error (NameError)": ("print(unknown_var)", ValueError),
        "Forbidden import": ("import os", ValueError),
        "Forbidden import from": ("from os import system", ValueError),
        "Forbidden call (eval)": ("eval('1+1')", ValueError),
        "Forbidden call (open)": ("open('test.txt', 'w')", ValueError),
        "Forbidden call (__import__)": ("__import__('os')", ValueError),
        "Forbidden del": ("x=10\ndel x", ValueError),
        "Accessing __builtins__ directly (allowed if item is whitelisted)": ("print(__builtins__.str(123))", "123\n"),
        "Accessing forbidden __builtins__ (e.g. open)": ("__builtins__.open('file.txt')", ValueError),
        "Empty snippet": ("", None),
        "Snippet with only comments": ("# This is a comment", None),
    }

    for description, (snippet, expected_outcome) in test_snippets.items():
        print(f"--- Testing: {description} ---")
        print(f"Snippet:\n{snippet}")
        try:
            result = run_python_snippet(snippet)
            print(f"Actual output:\n{result!r}") # Use !r to show None clearly and newlines
            if expected_outcome is ValueError:
                print("!!! TEST FAILED: Expected ValueError, but got a result.")
            elif result != expected_outcome:
                print(f"!!! TEST FAILED: Expected:\n{expected_outcome!r}")
        except ValueError as e:
            print(f"Caught ValueError (as expected for some tests): {e}")
            if expected_outcome is not ValueError:
                print(f"!!! TEST FAILED: Expected result '{expected_outcome!r}', but got ValueError.")
        except Exception as e:
            print(f"!!! TEST FAILED: Unexpected exception: {type(e).__name__}: {e}")
        print("-" * 30 + "\n")
