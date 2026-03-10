import ast
import io
import sys

# Define a list of disallowed built-in function names
DISALLOWED_BUILTINS = {
    'open',
    # Add other disallowed built-in functions here
}

# Define a list of disallowed module names
DISALLOWED_MODULES = {
    'socket',
    'os',
    'subprocess',
    'shutil',
    'ftplib',
    'http.client',
    'urllib',
    # Add other disallowed modules here
}

# Define a list of disallowed attribute calls (e.g., os.system)
DISALLOWED_ATTRIBUTES = {
    ('os', 'system'),
    ('os', 'spawn'),
    ('os', 'popen'),
    # Add other disallowed module.attribute calls
}

def analyze_code_safety(code_string: str) -> bool:
    """
    Analyzes a string of Python code for disallowed operations.

    Args:
        code_string: The Python code to analyze.

    Returns:
        True if the code is considered safe, False otherwise.
    """
    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        # If the code can't be parsed, it's not safe to execute
        return False

    for node in ast.walk(tree):
        # Check for disallowed function calls (e.g., open())
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in DISALLOWED_BUILTINS:
                    print(f"Disallowed built-in function call: {node.func.id}")
                    return False
            elif isinstance(node.func, ast.Attribute):
                # Check for disallowed attribute calls (e.g., os.system())
                # This requires checking the object being called (e.g., 'os' in 'os.system()')
                if isinstance(node.func.value, ast.Name):
                    module_name = node.func.value.id
                    attr_name = node.func.attr
                    if (module_name, attr_name) in DISALLOWED_ATTRIBUTES:
                        print(f"Disallowed attribute call: {module_name}.{attr_name}")
                        return False

        # Check for disallowed imports (e.g., import socket)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in DISALLOWED_MODULES:
                    print(f"Disallowed import: {alias.name}")
                    return False
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module in DISALLOWED_MODULES:
                print(f"Disallowed import from: {node.module}")
                return False
            # Could also check node.names for specific dangerous imports from allowed modules

    return True


def execute_safe_code(code_string: str) -> str | None:
    """
    Executes a string of Python code and captures its output.

    Args:
        code_string: The Python code to execute.

    Returns:
        The output produced by the code, or None if no output.
    """
    # Create a restricted global environment
    # Allow only a very limited set of built-ins for safety.
    # 'print' is essential for capturing output.
    # '__builtins__' can be a dictionary or a module.
    # Providing a dictionary is more explicit and controlled.
    allowed_builtins = {
        'print': print,
        'len': len,
        'str': str,
        'int': int,
        'float': float,
        'list': list,
        'dict': dict,
        'tuple': tuple,
        'range': range,
        'abs': abs,
        'min': min,
        'max': max,
        'sum': sum,
        'sorted': sorted,
        # Add any other safe built-ins you want to allow
    }
    restricted_globals = {"__builtins__": allowed_builtins}
    restricted_locals = {}

    # Redirect stdout to capture print statements
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    try:
        exec(code_string, restricted_globals, restricted_locals)
        output = captured_output.getvalue()
        return output if output else None
    except Exception as e:
        # Optionally, log the exception or handle it as needed
        # For now, if execution fails, we treat it as no output or an error state
        print(f"Error during execution: {e}")
        return f"Error: {e}" # Or return None, or raise a custom exception
    finally:
        # Restore stdout
        sys.stdout = old_stdout

if __name__ == '__main__':
    safe_code_1 = "print('Hello, world!')"
    safe_code_2 = "x = 1 + 2\nprint(x)"
    safe_code_3 = "import math\nprint(math.sqrt(4))"

    unsafe_code_1 = "open('some_file.txt', 'w')"
    unsafe_code_2 = "import socket\ns = socket.socket()"
    unsafe_code_3 = "import os\nos.system('echo unsafe')"
    unsafe_code_4 = "eval('print(\"unsafe eval\")')" # eval is a builtin, let's add it
    unsafe_code_5 = "__import__('os').system('echo unsafe')" # __import__ is a builtin

    DISALLOWED_BUILTINS.add('eval')
    DISALLOWED_BUILTINS.add('exec')
    DISALLOWED_BUILTINS.add('__import__')


    print(f"'{safe_code_1}' is safe: {analyze_code_safety(safe_code_1)}")
    print(f"'{safe_code_2}' is safe: {analyze_code_safety(safe_code_2)}")
    print(f"'{safe_code_3}' is safe: {analyze_code_safety(safe_code_3)}")

    print(f"'{unsafe_code_1}' is safe: {analyze_code_safety(unsafe_code_1)}")
    print(f"'{unsafe_code_2}' is safe: {analyze_code_safety(unsafe_code_2)}")
    print(f"'{unsafe_code_3}' is safe: {analyze_code_safety(unsafe_code_3)}")
    print(f"'{unsafe_code_4}' is safe: {analyze_code_safety(unsafe_code_4)}")
    print(f"'{unsafe_code_5}' is safe: {analyze_code_safety(unsafe_code_5)}")

    # Example of code that is syntactically incorrect
    invalid_code = "print('Hello'"
    print(f"'{invalid_code}' is safe: {analyze_code_safety(invalid_code)}")

    print("\n--- Testing execute_safe_code ---")
    code_to_execute_1 = "print('This is a test execution.')"
    code_to_execute_2 = "x = 10\ny = 20\nprint(x + y)"
    code_to_execute_3 = "a = [1, 2, 3]\nprint(len(a))" # Uses allowed 'len'
    code_no_output = "z = 100"
    # This code is "safe" by analyze_code_safety but uses a disallowed builtin for exec
    # execute_safe_code's restricted_globals will prevent 'open' from being available.
    code_with_disallowed_exec_builtin = "open('test.txt', 'w')"


    if analyze_code_safety(code_to_execute_1):
        print(f"Executing: {code_to_execute_1}")
        output = execute_safe_code(code_to_execute_1)
        print(f"Output:\n{output}")
    else:
        print(f"Skipping execution (unsafe): {code_to_execute_1}")

    if analyze_code_safety(code_to_execute_2):
        print(f"Executing: {code_to_execute_2}")
        output = execute_safe_code(code_to_execute_2)
        print(f"Output:\n{output}")
    else:
        print(f"Skipping execution (unsafe): {code_to_execute_2}")

    if analyze_code_safety(code_to_execute_3):
        print(f"Executing: {code_to_execute_3}")
        output = execute_safe_code(code_to_execute_3)
        print(f"Output:\n{output}")
    else:
        print(f"Skipping execution (unsafe): {code_to_execute_3}")

    if analyze_code_safety(code_no_output):
        print(f"Executing: {code_no_output}")
        output = execute_safe_code(code_no_output)
        print(f"Output: {output}") # Expected: None
    else:
        print(f"Skipping execution (unsafe): {code_no_output}")

    # Test execution of code that tries to use a disallowed builtin during exec
    # analyze_code_safety might pass it if 'open' is not in DISALLOWED_BUILTINS
    # but execute_safe_code should fail it due to restricted_globals.
    # Let's ensure 'open' is in DISALLOWED_BUILTINS for analyze_code_safety
    if 'open' not in DISALLOWED_BUILTINS:
        DISALLOWED_BUILTINS.add('open') # Ensure it's checked by analyze_code_safety

    print(f"Analyzing for safety: {code_with_disallowed_exec_builtin}")
    if analyze_code_safety(code_with_disallowed_exec_builtin):
        print(f"Executing: {code_with_disallowed_exec_builtin}")
        output = execute_safe_code(code_with_disallowed_exec_builtin)
        print(f"Output:\n{output}") # Expected: Error: name 'open' is not defined
    else:
        print(f"Skipping execution (unsafe by analyze_code_safety): {code_with_disallowed_exec_builtin}")

    unsafe_code_for_exec = "import os; os.system('echo pwned')"
    print(f"Analyzing for safety: {unsafe_code_for_exec}")
    if analyze_code_safety(unsafe_code_for_exec):
        print(f"Executing: {unsafe_code_for_exec}")
        output = execute_safe_code(unsafe_code_for_exec)
        print(f"Output:\n{output}")
    else:
        print(f"Skipping execution (unsafe by analyze_code_safety): {unsafe_code_for_exec}")
