import ast
import io
import contextlib

class CodeSafetyVisitor(ast.NodeVisitor):
    """
    AST visitor to check for prohibited operations in user scripts.
    Raises ValueError if unsafe operations are detected.
    """

    def visit_Import(self, node):
        """Disallow 'import ...' statements."""
        raise ValueError("Imports are not allowed.")

    def visit_ImportFrom(self, node):
        """Disallow 'from ... import ...' statements."""
        raise ValueError("Imports are not allowed.")

    def visit_Attribute(self, node):
        """
        Disallow access to attributes starting with an underscore.
        This is a broad check to prevent access to private/special/dunder
        attributes, which are common vectors for sandbox escapes (e.g.,
        accessing __globals__, __class__, __subclasses__, etc.).
        """
        if node.attr.startswith('_'):
            raise ValueError(
                f"Access to attributes starting with '_' (e.g., '{node.attr}') is not allowed."
            )
        # Continue visiting other parts of the AST, like the object
        # the attribute is being accessed on.
        self.generic_visit(node)


def _validate_script_safety(run_code: str) -> ast.AST:
    """
    Parses the script and validates its AST for prohibited operations.

    Args:
        run_code: The Python script string.

    Returns:
        The parsed and validated AST.

    Raises:
        ValueError: If syntax error or prohibited operations are found.
    """
    try:
        tree = ast.parse(run_code)
    except SyntaxError as e:
        raise ValueError(f"Syntax error in script: {e}")

    visitor = CodeSafetyVisitor()
    # The visitor.visit(tree) call can raise ValueError directly.
    # No need for a try/except here if we let it propagate.
    visitor.visit(tree)
    return tree


def _execute_validated_script(tree: ast.AST) -> str | None:
    """
    Executes a validated AST in a restricted environment and captures stdout.

    Args:
        tree: The validated AST of the script.

    Returns:
        The captured stdout of the script, or None if no output.

    Raises:
        ValueError: If any error occurs during script execution.
    """
    allowed_builtins = {
        'print': print,
        'len': len,
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
        'list': list,
        'dict': dict,
        'tuple': tuple,
        'set': set,
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'sorted': sorted,
        'range': range,
        'zip': zip,
        'enumerate': enumerate,
        'reversed': reversed,
        'all': all,
        'any': any,
        'isinstance': isinstance,
        'callable': callable,
        'hash': hash,
        'repr': repr,
        'ascii': ascii,
        'format': format,
        'pow': pow,
        'divmod': divmod,
        'True': True,
        'False': False,
        'None': None,
        'Exception': Exception,
        'ValueError': ValueError,
        'TypeError': TypeError,
        'IndexError': IndexError,
        'KeyError': KeyError,
        'AttributeError': AttributeError,
        'NameError': NameError,
        'ZeroDivisionError': ZeroDivisionError,
        'globals': lambda: {},
        'locals': lambda: {},
        'vars': lambda: {},
        'dir': lambda x=None: [], # type: ignore
    }

    restricted_globals = {"__builtins__": allowed_builtins}
    restricted_locals = {}

    stdout_capture = io.StringIO()
    try:
        compiled_code = compile(tree, filename="<user_script>", mode="exec")
        with contextlib.redirect_stdout(stdout_capture):
            exec(compiled_code, restricted_globals, restricted_locals)
    except Exception as e:
        raise ValueError(f"Error during script execution: {type(e).__name__}: {e}")

    output = stdout_capture.getvalue()
    return output if output else None


def restricted_run_code(run_code: str):
    """
    Executes a Python script specified by the user, but limits the script
    to operations considered safe.

    Args:
        run_code: A string containing the user's Python script.

    Returns:
        The output of the script (stdout), or None if no output.

    Raises:
        ValueError: If the script attempts prohibited actions, has syntax issues,
                    or encounters an error during execution in the restricted environment.
    """
    # _validate_script_safety can raise ValueError (syntax, safety)
    # _execute_validated_script can raise ValueError (runtime)
    # These ValueErrors are propagated as per the function's docstring.
    tree = _validate_script_safety(run_code)
    output = _execute_validated_script(tree)
    return output


if __name__ == '__main__':
    # Example Usage (for testing purposes)
    print("--- Test Safe Script ---")
    safe_code = """
a = 10
b = 20
print(a + b)
my_list = [1, 2, 3]
print(len(my_list))
for i in range(3):
    print(f"Number: {i}")
"""
    try:
        result = restricted_run_code(safe_code)
        print("Output:\n", result if result else "No output")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Test Script with Syntax Error ---")
    syntax_error_code = "print('Hello'"
    try:
        result = restricted_run_code(syntax_error_code)
        print("Output:\n", result if result else "No output")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Test Script with Import (Prohibited) ---")
    import_code = "import os"
    try:
        result = restricted_run_code(import_code)
        print("Output:\n", result if result else "No output")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Test Script with Underscore Attribute Access (Prohibited) ---")
    underscore_attr_code = "s = 'hello'\nprint(s.__class__)"
    try:
        result = restricted_run_code(underscore_attr_code)
        print("Output:\n", result if result else "No output")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Test Script with Prohibited Builtin Call (e.g., open) ---")
    prohibited_builtin_code = "f = open('test.txt', 'w')"
    try:
        result = restricted_run_code(prohibited_builtin_code)
        print("Output:\n", result if result else "No output")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Test Script with Runtime Error (e.g., ZeroDivisionError) ---")
    runtime_error_code = "print(1 / 0)"
    try:
        result = restricted_run_code(runtime_error_code)
        print("Output:\n", result if result else "No output")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Test Script that is just an expression ---")
    expression_code = "10 + 5" # No print, so no stdout
    try:
        result = restricted_run_code(expression_code)
        print("Output:\n", result if result else "No output") # Expected: No output
    except ValueError as e:
        print(f"Error: {e}")
    
    print("\n--- Test Script using allowed dunder (e.g. for operators, implicitly) ---")
    # This is fine because it's operator overloading, not direct access to __add__
    # Direct access like `(1).__add__(2)` would be `AttributeError` because int has no `__add__` attr
    # or if it was `obj.__add__` it would be caught by `visit_Attribute` if `__add__` starts with `_`.
    # The `startswith('_')` check in `visit_Attribute` would block `print((1).__add__(2))` if `__add__` was an attribute.
    # However, `1 + 2` itself doesn't involve an `ast.Attribute` node for `__add__`.
    allowed_dunder_code = "print(1 + 2)"
    try:
        result = restricted_run_code(allowed_dunder_code)
        print("Output:\n", result if result else "No output")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Test Script trying to access __builtins__ via globals() ---")
    access_globals_code = "print(globals()['__builtins__'])"
    try:
        result = restricted_run_code(access_globals_code) # globals() returns {}
        print("Output:\n", result if result else "No output")
    except ValueError as e:
        print(f"Error: {e}") # Expected: Error during script execution: KeyError: '__builtins__'
        
    print("\n--- Test Script trying to use __import__ ---")
    access_import_code = "__import__('os')"
    try:
        result = restricted_run_code(access_import_code)
        print("Output:\n", result if result else "No output")
    except ValueError as e:
        print(f"Error: {e}") # Expected: NameError: name '__import__' is not defined
