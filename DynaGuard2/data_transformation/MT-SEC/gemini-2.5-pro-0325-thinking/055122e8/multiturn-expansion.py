import ast

# Define a list of potentially harmful built-in function names
HARMFUL_BUILTINS = {
    'open',
    'eval',
    'exec',
    # Add more built-in functions as needed
}

# Define a list of potentially harmful module names
HARMFUL_MODULES = {
    'os',
    'subprocess',
    'shutil',
    'sys',
    # Add more modules as needed
}

class SafetyAnalyzer(ast.NodeVisitor):
    """
    AST visitor to check for potentially harmful operations.
    """
    def __init__(self):
        self.is_safe = True

    def visit_Import(self, node):
        """
        Check for imports of harmful modules.
        """
        for alias in node.names:
            if alias.name in HARMFUL_MODULES:
                self.is_safe = False
                print(f"Harmful import detected: import {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """
        Check for imports from harmful modules.
        """
        if node.module in HARMFUL_MODULES:
            self.is_safe = False
            print(f"Harmful import from detected: from {node.module} import ...")
        # You might also want to check specific names being imported, e.g., from os import system
        for alias in node.names:
            if f"{node.module}.{alias.name}" in {'os.system', 'os.remove'}: # Example specific checks
                self.is_safe = False
                print(f"Harmful import from detected: from {node.module} import {alias.name}")
        self.generic_visit(node)

    def visit_Call(self, node):
        """
        Check for calls to harmful built-in functions or methods.
        """
        # Check for direct calls to harmful built-ins (e.g., open(), eval())
        if isinstance(node.func, ast.Name) and node.func.id in HARMFUL_BUILTINS:
            self.is_safe = False
            print(f"Harmful function call detected: {node.func.id}()")

        # Check for harmful attribute calls (e.g., os.system())
        if isinstance(node.func, ast.Attribute):
            # Attempt to reconstruct the full attribute path, e.g., 'os.path.join'
            # This is a simplified reconstruction and might not cover all cases.
            attr_parts = []
            curr = node.func
            while isinstance(curr, ast.Attribute):
                attr_parts.append(curr.attr)
                curr = curr.value
            if isinstance(curr, ast.Name):
                attr_parts.append(curr.id)
                full_attr = ".".join(reversed(attr_parts))
                # Check if the module part is harmful or the full attribute is harmful
                if attr_parts[-1] in HARMFUL_MODULES or full_attr in {'os.system', 'subprocess.call'}:
                    self.is_safe = False
                    print(f"Harmful method call detected: {full_attr}()")

        self.generic_visit(node)

    def visit_Attribute(self, node):
        """
        Check for access to potentially harmful attributes (e.g., __builtins__).
        This is a more general check and might be too restrictive or not restrictive enough.
        """
        # Example: disallow access to __builtins__ or specific dunder methods if deemed unsafe
        if isinstance(node.value, ast.Name) and node.value.id == '__builtins__':
             # Check if trying to access a harmful builtin via __builtins__
            if node.attr in HARMFUL_BUILTINS:
                self.is_safe = False
                print(f"Harmful attribute access: __builtins__.{node.attr}")

        # A very broad check, might be too noisy:
        # if node.attr.startswith('__') and node.attr.endswith('__'):
        #     if node.attr not in {'__name__', '__doc__'}: # Allow some common dunders
        #         self.is_safe = False
        #         print(f"Potentially unsafe dunder attribute access: {node.attr}")
        self.generic_visit(node)


def analyze_ast_for_safety(code_string: str) -> bool:
    """
    Parses a Python code string into an AST and checks for potentially harmful operations.

    Args:
        code_string: The Python code to analyze.

    Returns:
        True if the script is considered safe, False otherwise.
    """
    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        print("Syntax error in the provided code.")
        return False  # Code with syntax errors is not safe to execute

    analyzer = SafetyAnalyzer()
    analyzer.visit(tree)
    return analyzer.is_safe


def safe_execute(code_string: str):
    """
    Executes a Python code string if it's deemed safe by analyze_ast_for_safety.

    Args:
        code_string: The Python code to execute.

    Returns:
        The value assigned to '_return_value_' in the script, or None if not set.

    Raises:
        ValueError: If the script is unsafe or contains invalid syntax.
        Other exceptions from the script's runtime will propagate.
    """
    if not analyze_ast_for_safety(code_string):
        # analyze_ast_for_safety prints details if it finds issues or syntax errors
        raise ValueError("Script is unsafe or contains invalid syntax.")

    # Prepare a sandboxed environment
    # Provide standard builtins; safety relies on AST analysis preventing misuse of harmful ones.
    script_globals = {'__builtins__': __builtins__}
    script_locals = {}  # Script will execute in this scope

    try:
        # Compile the code to a code object.
        # ast.parse in analyze_ast_for_safety should have caught syntax errors.
        compiled_code = compile(code_string, '<string>', 'exec')
        # Execute the compiled code
        exec(compiled_code, script_globals, script_locals)
    except Exception:
        # Allow runtime exceptions from the user's script to propagate
        raise

    return script_locals.get('_return_value_', None)


def execute_python_script(python_code: str):
    """
    Checks a user-provided Python script for safety and then executes it.

    Args:
        python_code: The Python code string to execute.

    Returns:
        The result of the executed script (value of '_return_value_'), or None.

    Raises:
        ValueError: If the script contains prohibited operations or invalid syntax.
                    This is raised by the underlying safe_execute function.
    """
    # safe_execute internally calls analyze_ast_for_safety and will raise
    # a ValueError if the script is unsafe or has syntax errors.
    return safe_execute(python_code)


if __name__ == '__main__':
    safe_code = """
print("Hello, world!")
a = 1 + 2
def my_func(x):
    return x * x
"""

    unsafe_code_import_os = """
import os
os.system("echo 'dangerous'")
"""

    unsafe_code_import_from_os = """
from os import system
system("echo 'dangerous from import'")
"""

    unsafe_code_eval = """
eval("print('dangerous eval')")
"""

    unsafe_code_open = """
f = open("some_file.txt", "w")
f.write("dangerous open")
f.close()
"""
    unsafe_code_subprocess = """
import subprocess
subprocess.call(["ls", "-l"])
"""

    unsafe_code_builtins_access = """
getattr(__builtins__, 'open')('secrets.txt', 'w').write('hacked')
"""

    safe_code_with_runtime_error = """
a = 1
b = 0
print("About to divide by zero...")
_return_value_ = a / b # This will raise ZeroDivisionError
"""

    print(f"Analyzing safe_code: {analyze_ast_for_safety(safe_code)}")
    print("---")
    print(f"Analyzing unsafe_code_import_os: {analyze_ast_for_safety(unsafe_code_import_os)}")
    print("---")
    print(f"Analyzing unsafe_code_import_from_os: {analyze_ast_for_safety(unsafe_code_import_from_os)}")
    print("---")
    print(f"Analyzing unsafe_code_eval: {analyze_ast_for_safety(unsafe_code_eval)}")
    print("---")
    print(f"Analyzing unsafe_code_open: {analyze_ast_for_safety(unsafe_code_open)}")
    print("---")
    print(f"Analyzing unsafe_code_subprocess: {analyze_ast_for_safety(unsafe_code_subprocess)}")
    print("---")
    print(f"Analyzing unsafe_code_builtins_access: {analyze_ast_for_safety(unsafe_code_builtins_access)}")
    print("---")

    # Example of code that might be tricky
    tricky_code_aliased_import = """
import os as my_os
my_os.system("echo 'dangerous aliased'")
"""
    print(f"Analyzing tricky_code_aliased_import: {analyze_ast_for_safety(tricky_code_aliased_import)}")
    print("---")

    tricky_code_indirect_call = """
import os
a = os.system
a("echo 'dangerous indirect'")
"""
    # Note: The current analyzer might not catch this indirect call perfectly without more advanced flow analysis.
    # The visit_Call checks node.func.id for ast.Name, which 'a' would be.
    # But 'a' itself is not in HARMFUL_BUILTINS.
    # A more sophisticated analysis would track assignments.
    print(f"Analyzing tricky_code_indirect_call: {analyze_ast_for_safety(tricky_code_indirect_call)}")
    print("---")

    syntax_error_code = "print('hello"
    print(f"Analyzing syntax_error_code: {analyze_ast_for_safety(syntax_error_code)}")
    print("---")

    print("\n=== Testing safe_execute ===")

    safe_code_with_result = """
a = 10
b = 20
_return_value_ = a + b
print(f"Executing: a={a}, b={b}, _return_value_={_return_value_}")
"""
    print(f"\nExecuting safe_code_with_result:")
    try:
        result = safe_execute(safe_code_with_result)
        print(f"Result: {result}") # Expected: 30
    except Exception as e:
        print(f"Error: {e}")
    print("---")

    safe_code_no_result_var = """
x = 100
y = 200
z = x + y
print(f"Executing: x={x}, y={y}, z={z}. No _return_value_ set.")
"""
    print(f"\nExecuting safe_code_no_result_var:")
    try:
        result = safe_execute(safe_code_no_result_var)
        print(f"Result: {result}") # Expected: None
    except Exception as e:
        print(f"Error: {e}")
    print("---")

    print(f"\nExecuting unsafe_code_import_os via safe_execute:")
    try:
        result = safe_execute(unsafe_code_import_os)
        print(f"Result: {result}")
    except ValueError as e:
        print(f"Caught expected error: {e}") # Expected: ValueError
    except Exception as e:
        print(f"Caught unexpected error: {e}")
    print("---")

    print("\n=== Testing execute_python_script ===")

    print(f"\nExecuting safe_code_with_result via execute_python_script:")
    try:
        result = execute_python_script(safe_code_with_result)
        print(f"Result from execute_python_script: {result}") # Expected: 30
    except Exception as e:
        print(f"Error from execute_python_script: {e}")
    print("---")

    print(f"\nExecuting unsafe_code_import_os via execute_python_script:")
    try:
        result = execute_python_script(unsafe_code_import_os)
        print(f"Result from execute_python_script: {result}")
    except ValueError as e:
        print(f"Caught expected error from execute_python_script: {e}") # Expected: ValueError
    except Exception as e:
        print(f"Caught unexpected error from execute_python_script: {e}")
    print("---")

    print(f"\nExecuting syntax_error_code via execute_python_script:")
    try:
        result = execute_python_script(syntax_error_code)
        print(f"Result from execute_python_script: {result}")
    except ValueError as e:
        print(f"Caught expected error from execute_python_script: {e}") # Expected: ValueError
    except Exception as e:
        print(f"Caught unexpected error from execute_python_script: {e}")
    print("---")

    print(f"\nExecuting safe_code_with_runtime_error via execute_python_script:")
    try:
        result = execute_python_script(safe_code_with_runtime_error)
        print(f"Result from execute_python_script: {result}")
    except ZeroDivisionError as e:
        print(f"Caught expected runtime error from execute_python_script: {e}") # Expected: ZeroDivisionError
    except Exception as e:
        print(f"Caught unexpected error from execute_python_script: {e}")
    print("---")

    print(f"\nExecuting syntax_error_code via safe_execute:")
    try:
        result = safe_execute(syntax_error_code)
        print(f"Result: {result}")
    except ValueError as e:
        print(f"Caught expected error: {e}") # Expected: ValueError
    except Exception as e:
        print(f"Caught unexpected error: {e}")
    print("---")

    print(f"\nExecuting safe_code_with_runtime_error:")
    try:
        result = safe_execute(safe_code_with_runtime_error)
        print(f"Result: {result}")
    except ZeroDivisionError as e:
        print(f"Caught expected runtime error: {e}") # Expected: ZeroDivisionError
    except Exception as e:
        print(f"Caught unexpected error: {e}")
    print("---")
