import ast
import math
from typing import Any, Dict, Optional

# --- AST Safety Validation ---

# Set of function names that are disallowed to be called directly or as methods.
DISALLOWED_FUNCTION_CALLS = {
    'eval', 'exec', 'compile', '__import__',
    'open', 'exit', 'quit', 'help', 'input',
    'setattr', 'getattr', 'delattr',
    # Attributes that could lead to dangerous operations if called or accessed
    '__class__', '__subclasses__', '__bases__', '__mro__',
    '__globals__', '__code__', '__closure__', '__func__',
    '__self__', '__dict__',
    # System related - typically on modules like 'os', 'subprocess' which won't be available
    'system', 'spawn', 'popen', 'fork',
}

# Set of attribute names that are disallowed for direct access.
DISALLOWED_ATTR_NAMES = {
    '__class__', '__subclasses__', '__bases__', '__mro__',
    '__globals__', '__code__', '__closure__', '__func__',
    '__self__', '__dict__', '__builtins__', '__file__',
    '__package__', '__path__', '__spec__',
}


class AstSafetyValidator(ast.NodeVisitor):
    """
    Validates an AST tree to ensure it doesn't contain prohibited operations.
    Raises ValueError if an unsafe operation is detected.
    """
    def __init__(self):
        super().__init__()
        self._is_safe = True # Internal flag, though methods directly raise

    def _check_safety(self, condition, error_message):
        if not condition:
            self._is_safe = False
            raise ValueError(error_message)

    def visit_Import(self, node: ast.Import):
        self._check_safety(False, "Imports are not allowed.")

    def visit_ImportFrom(self, node: ast.ImportFrom):
        self._check_safety(False, "ImportFrom is not allowed.")

    def visit_Call(self, node: ast.Call):
        func_name_to_check = None
        if isinstance(node.func, ast.Name): # e.g., my_func()
            func_name_to_check = node.func.id
        elif isinstance(node.func, ast.Attribute): # e.g., obj.method() or math.sqrt()
            func_name_to_check = node.func.attr
            # Additionally, check if trying to call something like __builtins__.open()
            if isinstance(node.func.value, ast.Name) and node.func.value.id == '__builtins__':
                if node.func.attr in DISALLOWED_FUNCTION_CALLS:
                    self._check_safety(False, f"Call to disallowed built-in function '{node.func.attr}' is not allowed.")

        if func_name_to_check in DISALLOWED_FUNCTION_CALLS:
            self._check_safety(False, f"Call to disallowed function or method '{func_name_to_check}' is not allowed.")

        super().generic_visit(node) # Visit arguments and other parts of the call

    def visit_Attribute(self, node: ast.Attribute):
        # Check direct attribute access, e.g., my_obj.dangerous_attr
        if node.attr in DISALLOWED_ATTR_NAMES:
            self._check_safety(False, f"Access to disallowed attribute '{node.attr}' is not allowed.")
        
        # Prevent access like some_object.__builtins__
        if isinstance(node.value, ast.Name) and node.value.id == '__builtins__':
             self._check_safety(False, "Access to attributes of '__builtins__' is not allowed directly.")

        super().generic_visit(node)

    def visit_Name(self, node: ast.Name):
        # Prevent direct use of __builtins__ as a variable name if it implies accessing the raw builtins module
        if node.id == '__builtins__':
            self._check_safety(False, "Direct reference to '__builtins__' is not allowed.")
        super().generic_visit(node)

    def visit_Delete(self, node: ast.Delete):
        # Deleting items can be risky depending on the context (e.g., from a shared dict)
        # For simplicity in a restricted environment, disallow 'del' statements.
        self._check_safety(False, "'del' statements are not allowed.")


# --- Script Execution Function ---

def execute_python_script(python_code: str) -> Optional[Any]:
    """
    Executes a user-provided Python script string in a sandboxed environment.

    Args:
        python_code: The Python script as a string.

    Returns:
        The result of the last expression in the script, or None if the script
        does not end with an expression or has no specific result.

    Raises:
        ValueError: If the script contains invalid syntax, prohibited operations,
                    or if a runtime error occurs during execution.
    """
    # 1. Parse the code into an AST
    try:
        tree = ast.parse(python_code, filename='<user_script>')
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax: {e}") from e

    # 2. Validate the AST for prohibited operations
    validator = AstSafetyValidator()
    try:
        validator.visit(tree)
    except ValueError as e: # Catch specific errors raised by the validator
        # Re-raise as a clear "Prohibited operation" error
        raise ValueError(f"Prohibited operation detected: {e}") from e

    # 3. Prepare a safe execution environment
    safe_builtins: Dict[str, Any] = {
        'print': print,
        'abs': abs, 'min': min, 'max': max, 'round': round, 'sum': sum,
        'len': len, 'range': range, 'zip': zip, 'enumerate': enumerate,
        'str': str, 'int': int, 'float': float, 'bool': bool,
        'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
        'isinstance': isinstance, # Use with caution; types should be controlled
        'Exception': Exception, # Allow raising/catching generic exceptions
        # Basic error types
        'ValueError': ValueError, 'TypeError': TypeError, 'NameError': NameError,
        'IndexError': IndexError, 'KeyError': KeyError, 'ZeroDivisionError': ZeroDivisionError,
    }
    script_globals: Dict[str, Any] = {
        "__builtins__": safe_builtins,
        "math": math, # Allow using math.sqrt(), math.pi, etc.
    }
    script_locals: Dict[str, Any] = {} # For script's local variables

    # 4. Modify AST to capture the result of the last expression statement
    RESULT_VAR_NAME = "__script_result__"
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        # If the last statement is an expression, assign its value to RESULT_VAR_NAME
        assign_node = ast.Assign(
            targets=[ast.Name(id=RESULT_VAR_NAME, ctx=ast.Store())],
            value=tree.body[-1].value
        )
        # Important: Ensure new nodes have line/col info for compilation
        tree.body[-1] = ast.fix_missing_locations(assign_node)

    # 5. Compile the (potentially modified) AST
    try:
        compiled_code = compile(tree, filename='<user_script>', mode='exec')
    except Exception as e:
        raise ValueError(f"Error compiling script: {type(e).__name__} - {e}") from e

    # 6. Execute the compiled code
    try:
        exec(compiled_code, script_globals, script_locals)
    except Exception as e:
        # Catch runtime errors from the script execution
        raise ValueError(f"Error executing script: {type(e).__name__} - {e}") from e

    # 7. Return the captured result, or None
    return script_locals.get(RESULT_VAR_NAME)

if __name__ == '__main__':
    test_scripts = [
        # --- Success cases ---
        ("1 + 2", 3),
        ("x = 10\ny = 20\nx + y", 30),
        ("abs(-5.5)", 5.5),
        ("math.sqrt(16)", 4.0),
        ("sum([1, 2, 3])", 6),
        ("'hello ' + 'world'", "hello world"),
        ("x = 5", None), # Assignment, no final expression to return by default
        ("print('hello')\n42", 42), # Print then expression
        ("print('side effect')", None), # Only side effect
        ("len([1,2,3])", 3),
        ("def foo(x): return x*x\nfoo(10)", 100),
        ("def bar(x, y=1): return x + y\nbar(5, y=3)", 8),
        ("a = [1,2,3]\na.append(4)\na", [1,2,3,4]), # list methods are fine if list is safe
        ("d = {'a': 1}\nd['b'] = 2\nd", {'a':1, 'b':2}),
        ("[i*i for i in range(5) if i % 2 == 0]", [0, 4, 16]), # List comprehension
        ("math.pi", math.pi),
        ("10 / 2", 5.0),
        # --- Expected Error Cases ---
        # Syntax Errors
        ("1 +", ValueError("Invalid syntax: unexpected EOF while parsing (<user_script>, line 1)")),
        ("x = )", ValueError("Invalid syntax: invalid syntax (<user_script>, line 1)")),
        # Prohibited Operations
        ("import os", ValueError("Prohibited operation detected: Imports are not allowed.")),
        ("from math import pi", ValueError("Prohibited operation detected: ImportFrom is not allowed.")),
        ("eval('1+1')", ValueError("Prohibited operation detected: Call to disallowed function or method 'eval' is not allowed.")),
        ("open('file.txt')", ValueError("Prohibited operation detected: Call to disallowed function or method 'open' is not allowed.")),
        ("().__class__", ValueError("Prohibited operation detected: Access to disallowed attribute '__class__' is not allowed.")),
        ("x = {}\nx.__dict__", ValueError("Prohibited operation detected: Access to disallowed attribute '__dict__' is not allowed.")),
        ("del x", ValueError("Prohibited operation detected: 'del' statements are not allowed.")),
        # Runtime Errors (wrapped in ValueError by execute_python_script)
        ("1 / 0", ValueError("Error executing script: ZeroDivisionError - division by zero")),
        ("non_existent_var", ValueError("Error executing script: NameError - name 'non_existent_var' is not defined")),
        ("x = [1]\nx[5]", ValueError("Error executing script: IndexError - list index out of range")),
        ("d={}\nd['key']", ValueError("Error executing script: KeyError - 'key'")),
        ("math.sqrt(-1)", ValueError("Error executing script: ValueError - math domain error")), # math itself raises ValueError
        ("'a' + 1", ValueError("Error executing script: TypeError - can only concatenate str (not \"int\") to str")),
    ]

    for i, (script, expected_outcome) in enumerate(test_scripts):
        print(f"--- Test Case {i+1} ---")
        print(f"Script:\n{script}")
        try:
            result = execute_python_script(script)
            if isinstance(expected_outcome, ValueError):
                print(f"  Status: FAIL (Expected ValueError, but got result: {result!r})")
            elif result == expected_outcome:
                print(f"  Status: PASS (Result: {result!r})")
            # Handle float comparison with tolerance
            elif isinstance(result, float) and isinstance(expected_outcome, float) and math.isclose(result, expected_outcome):
                print(f"  Status: PASS (Result: {result!r} approx. {expected_outcome!r})")
            else:
                print(f"  Status: FAIL (Expected: {expected_outcome!r}, Got: {result!r})")

        except ValueError as e:
            if isinstance(expected_outcome, ValueError):
                # Compare error messages (or types if messages are too specific/dynamic)
                # For simplicity, checking if the caught error message starts with the expected one.
                if str(e).startswith(str(expected_outcome)):
                    print(f"  Status: PASS (Caught expected ValueError: {e})")
                else:
                    print(f"  Status: FAIL (Caught ValueError, but message mismatch.\n          Expected prefix: {expected_outcome}\n          Got: {e})")
            else:
                print(f"  Status: FAIL (Unexpected ValueError: {e}. Expected result: {expected_outcome!r})")
        except Exception as e:
            print(f"  Status: FAIL (Unexpected Exception: {type(e).__name__} - {e}. Expected: {expected_outcome!r})")
        print("-" * 20)

    # Example of a script that uses a loop and defines a function
    multiline_script_example = """
total = 0
for i in range(1, 6):  # 1 to 5
    total += i

def multiply(a, b):
    return a * b

result = multiply(total, 2)
# The last expression is 'result', so its value should be returned
result
"""
    print("--- Multiline Script Example ---")
    print(f"Script:\n{multiline_script_example.strip()}")
    try:
        res = execute_python_script(multiline_script_example)
        # sum(1..5) = 15. 15 * 2 = 30.
        if res == 30:
            print(f"  Status: PASS (Result: {res})")
        else:
            print(f"  Status: FAIL (Expected: 30, Got: {res})")
    except Exception as e:
        print(f"  Status: FAIL (Error: {e})")
    print("-" * 20)
