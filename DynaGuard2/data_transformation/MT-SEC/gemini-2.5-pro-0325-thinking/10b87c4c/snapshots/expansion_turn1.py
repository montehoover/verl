import ast

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
