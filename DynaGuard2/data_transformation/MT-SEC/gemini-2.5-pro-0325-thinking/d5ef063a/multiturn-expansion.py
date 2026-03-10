import ast
import io
import sys
import contextlib
import math # Ensure math is imported for ALLOWED_MODULES

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

# Whitelist of safe built-in functions and objects for the sandboxed environment
SAFE_BUILTINS = {
    'print': print,
    'len': len,
    'range': range,
    'abs': abs,
    'all': all,
    'any': any,
    'ascii': ascii,
    'bin': bin,
    'bool': bool,
    'bytearray': bytearray,
    'bytes': bytes,
    'callable': callable,
    'chr': chr,
    'complex': complex,
    'dict': dict,
    'divmod': divmod,
    'enumerate': enumerate,
    'filter': filter,
    'float': float,
    'format': format,
    'frozenset': frozenset,
    'getattr': getattr, # Use with caution, ensure global scope is minimal
    'hasattr': hasattr, # Use with caution
    'hash': hash,
    'hex': hex,
    'id': id,
    'int': int,
    'isinstance': isinstance,
    'issubclass': issubclass,
    'iter': iter,
    'list': list,
    'map': map,
    'max': max,
    'min': min,
    'next': next,
    'oct': oct,
    'ord': ord,
    'pow': pow,
    'repr': repr,
    'reversed': reversed,
    'round': round,
    'set': set,
    'slice': slice,
    'sorted': sorted,
    'str': str,
    'sum': sum,
    'super': super, # Class related, generally safe if class creation is limited
    'tuple': tuple,
    'type': type,
    'zip': zip,
    # Common exceptions (can be raised and caught by user code)
    'ArithmeticError': ArithmeticError,
    'AssertionError': AssertionError,
    'AttributeError': AttributeError,
    'Exception': Exception,
    'IndexError': IndexError,
    'KeyError': KeyError,
    'LookupError': LookupError,
    'MemoryError': MemoryError,
    'NameError': NameError,
    'OverflowError': OverflowError,
    'RuntimeError': RuntimeError,
    'SyntaxError': SyntaxError, # Though code is parsed before exec
    'TypeError': TypeError,
    'ValueError': ValueError,
    'ZeroDivisionError': ZeroDivisionError,
}

# Whitelist of allowed modules, pre-imported and injected into the sandbox global scope
ALLOWED_MODULES = {
    "math": math,
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

def restricted_run_code(code_string: str):
    """
    Parses, checks for safety, and executes a string of Python code in a restricted environment.

    Args:
        code_string: The Python code to execute.

    Returns:
        The standard output of the script as a string, or None if there's no output.

    Raises:
        ValueError: If the code has syntax errors or performs prohibited actions.
        Other exceptions from the executed code may also propagate.
    """
    # 1. Parse code string to AST
    try:
        ast_tree = parse_python_code(code_string)
    except ValueError:
        raise  # Propagate syntax error

    # 2. Check AST for prohibited actions
    try:
        check_for_prohibited_actions(ast_tree)
    except ValueError:
        raise  # Propagate prohibited action error

    # 3. Compile AST to code object
    # <user_code> is a common convention for the filename of executed strings
    try:
        code_obj = compile(ast_tree, filename="<user_code>", mode="exec")
    except Exception as e:
        # This might happen for ASTs that parse but don't compile, though rare.
        raise ValueError(f"Error compiling AST: {e}")

    # 4. Prepare execution environment and execute
    # User code will have access to keys in ALLOWED_MODULES and SAFE_BUILTINS.
    # __builtins__ key is special: its value (a dict) is used for lookup of builtins.
    exec_globals = {
        **ALLOWED_MODULES,
        "__builtins__": SAFE_BUILTINS
    }
    
    output_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code_obj, exec_globals)
    except Exception:
        # Runtime errors from the user's code (e.g., ZeroDivisionError, NameError for undefined var)
        # will propagate. This is often desired.
        raise

    output = output_buffer.getvalue()
    return output if output else None

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
        "prohibited_shutil_rmtree_call": "import shutil\nshutil.rmtree('/tmp/dummy')", # Caught by PROHIBITED_IMPORTS
        "direct_os_system_if_os_available": "os.system('echo unsafe')", # Caught by PROHIBITED_METHOD_CALLS
        "safe_code_with_math_import_fails_later": "import math\nprint(math.sqrt(4))" # AST check passes, but fails at runtime due to no __import__
    }

    print("--- Testing parse_python_code and check_for_prohibited_actions ---")
    for name, code in test_cases.items():
        print(f"\n--- Test (parse/check): {name} ---")
        print(f"Code:\n{code}\n")
        ast_tree = None
        try:
            print("1. Parsing code...")
            ast_tree = parse_python_code(code)
            print("AST generated successfully.")
            # print(ast.dump(ast_tree, indent=2)) # Optional: print AST
            
            print("\n2. Checking for prohibited actions...")
            if check_for_prohibited_actions(ast_tree):
                print("AST check: Code is safe.")
        except ValueError as e:
            print(f"AST or Safety Check Error: {e}")
        print("--- End Test (parse/check) ---")

    # Test cases for restricted_run_code
    # Format: test_name: (code_string, expected_output_or_exception_type)
    run_test_cases = {
        "simple_print": ("print('Hello from sandbox')", "Hello from sandbox\n"),
        "math_usage_direct": ("print(math.sqrt(16))", "4.0\n"), # math is pre-injected
        "import_math_runtime_fail": ("import math\nprint(math.sqrt(16))", NameError), # Fails due to no __import__ in SAFE_BUILTINS
        "no_output_code": ("a = 1 + 1\nb=a*2", None),
        "multi_line_print": ("print('line1')\nprint('line2')", "line1\nline2\n"),
        "loop_print": ("for i in range(3):\n  print(i)", "0\n1\n2\n"),
        "runtime_zerodivision_error": ("print(1/0)", ZeroDivisionError),
        "runtime_name_error": ("print(undefined_variable)", NameError),
        "unsafe_import_os_ast_check": ("import os\nprint('should not run')", ValueError), # Caught by check_for_prohibited_actions
        "unsafe_builtin_open_ast_check": ("open('file.txt')", ValueError), # Caught by check_for_prohibited_actions
        "unsafe_direct_os_system_ast_check": ("os.system('echo unsafe')", ValueError), # Caught by check_for_prohibited_actions (method call)
        "syntax_error_runtime": ("print 'hello'", ValueError), # Caught by parse_python_code
        "empty_code": ("", None),
        "code_with_comments": ("# This is a comment\nprint('ok')", "ok\n"),
        "access_allowed_builtin_directly": ("print(str(123))", "123\n"),
        "try_disallowed_builtin_runtime_fail": ("__import__('os')", NameError), # __import__ not in SAFE_BUILTINS
    }

    print("\n\n--- Testing restricted_run_code ---")
    for name, (code, expected_result) in run_test_cases.items():
        print(f"\n--- Test (run): {name} ---")
        print(f"Code:\n{code}\n")
        try:
            output = restricted_run_code(code_string=code)
            print(f"Output: {repr(output)}") # Use repr to show None vs empty string clearly
            if isinstance(expected_result, type) and issubclass(expected_result, Exception):
                print(f"Error: Expected exception {expected_result.__name__}, but got output.")
                assert False, f"Test {name} failed: Expected exception {expected_result.__name__}"
            else:
                assert output == expected_result, f"Test {name} failed: Expected {repr(expected_result)}, got {repr(output)}"
                print("Result: Matches expected output.")
        except Exception as e:
            print(f"Caught Exception: {type(e).__name__}: {e}")
            if isinstance(expected_result, type) and issubclass(expected_result, Exception):
                assert isinstance(e, expected_result), f"Test {name} failed: Expected exception {expected_result.__name__}, got {type(e).__name__}"
                print(f"Result: Correctly caught {expected_result.__name__}.")
            else:
                print(f"Error: Expected output {repr(expected_result)}, but got exception {type(e).__name__}.")
                assert False, f"Test {name} failed: Expected output {repr(expected_result)}, got exception {type(e).__name__}"
        print("--- End Test (run) ---")
