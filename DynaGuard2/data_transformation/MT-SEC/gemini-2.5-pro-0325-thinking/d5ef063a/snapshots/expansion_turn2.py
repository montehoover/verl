import ast

PROHIBITED_BUILTIN_CALLS = {'open', 'eval', 'exec', '__import__'}
PROHIBITED_IMPORTS = {
    'os', 'sys', 'subprocess', 'shutil', 'socket', 'requests', 'urllib',
    'ftplib', 'telnetlib', 'http', 'asyncio', 'multiprocessing',
    'threading', 'ctypes', 'cgi', 'pickle', 'shelve'
}
# For calls like os.system()
PROHIBITED_METHOD_CALLS = {
    'os': {
        'system', 'remove', 'unlink', 'rmdir', 'mkdir', 'makedirs', 'listdir',
        'scandir', 'popen', 'spawnl', 'spawnle', 'spawnlp', 'spawnlpe',
        'spawnv', 'spawnve', 'spawnvp', 'spawnvpe', 'fork', 'execl', 'execle',
        'execlp', 'execlpe', 'execv', 'execve', 'execvp', 'execvpe', 'kill',
        'putenv', 'unsetenv', 'chdir', 'fchdir', 'chroot', 'startfile'
    },
    'subprocess': {'call', 'run', 'check_call', 'check_output', 'Popen'},
    'shutil': {
        'copy', 'copy2', 'copyfile', 'copyfileobj', 'copymode', 'copystat',
        'copytree', 'disk_usage', 'ignore_patterns', 'make_archive',
        'move', 'rmtree', 'unpack_archive', 'which'
    },
}

def parse_python_code(code_string: str):
    """
    Parses a string of Python code and returns its AST.

    Args:
        code_string: The Python code to parse.

    Returns:
        An AST object if the code is syntactically correct.

    Raises:
        ValueError: If the code has syntax errors.
    """
    try:
        tree = ast.parse(code_string)
        return tree
    except SyntaxError as e:
        raise ValueError(f"Syntax error in Python code: {e}")

def check_for_prohibited_actions(tree: ast.AST) -> bool:
    """
    Checks an AST for prohibited actions like file I/O or unsafe module usage.

    Args:
        tree: The AST to check.

    Returns:
        True if the code is safe.

    Raises:
        ValueError: If any prohibited action is found.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check for prohibited built-in function calls
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in PROHIBITED_BUILTIN_CALLS:
                    raise ValueError(
                        f"Prohibited built-in function call: {func_name} at line {getattr(node, 'lineno', 'unknown')}"
                    )
            # Check for prohibited method calls on modules (e.g., os.system())
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    module_name = node.func.value.id
                    method_name = node.func.attr
                    if module_name in PROHIBITED_METHOD_CALLS and \
                       method_name in PROHIBITED_METHOD_CALLS[module_name]:
                        raise ValueError(
                            f"Prohibited method call: {module_name}.{method_name} at line {getattr(node, 'lineno', 'unknown')}"
                        )
        
        # Check for prohibited module imports
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in PROHIBITED_IMPORTS:
                    raise ValueError(
                        f"Prohibited module import: {alias.name} at line {getattr(node, 'lineno', 'unknown')}"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module in PROHIBITED_IMPORTS:
                raise ValueError(
                    f"Prohibited module import (from ... import): {node.module} at line {getattr(node, 'lineno', 'unknown')}"
                )
    return True

if __name__ == '__main__':
    test_cases = {
        "valid_code": "print('Hello, world!')\na = 1 + 2",
        "invalid_syntax_code": "print 'Hello, world!'", # Python 2 syntax
        "more_invalid_syntax_code": "a = 1 +",
        "prohibited_open_call": "f = open('file.txt', 'w')",
        "prohibited_eval_call": "eval('1+1')",
        "prohibited_exec_call": "exec('print(1)')",
        "prohibited_import_os": "import os",
        "prohibited_import_os_as": "import os as my_os",
        "prohibited_from_import_os": "from os import system",
        "prohibited_from_import_sys": "from sys import exit",
        "prohibited_os_system_call": "import os\nos.system('echo unsafe')", # Caught by import
        "prohibited_subprocess_run_call": "import subprocess\nsubprocess.run(['ls'])", # Caught by import
        "prohibited_shutil_rmtree_call": "import shutil\nshutil.rmtree('/tmp/dummy')", # Caught by import
        "direct_os_system_if_os_available": "os.system('echo unsafe')", # Test if 'os' was pre-available
        "safe_code_with_math": "import math\nprint(math.sqrt(4))"
    }

    for name, code in test_cases.items():
        print(f"\n--- Testing: {name} ---")
        print(f"Code:\n{code}\n")
        try:
            print("1. Parsing code...")
            ast_tree = parse_python_code(code)
            print("AST generated successfully.")
            # print(ast.dump(ast_tree, indent=2)) # Optional: print AST
            
            print("\n2. Checking for prohibited actions...")
            if check_for_prohibited_actions(ast_tree):
                print("Code is safe.")
        except ValueError as e:
            print(f"Error: {e}")
        print("--- End Test ---")
