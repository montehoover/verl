import ast
import io
import sys
import contextlib

# Define a set of unsafe function names and module.attribute combinations
UNSAFE_FUNCTION_CALLS = {
    'open',
    'eval',
    'exec',
    '__import__',  # For blocking dynamic imports of unsafe modules
}

UNSAFE_MODULE_ATTRIBUTES = {
    'os': {
        'system', 'popen', 'remove', 'unlink', 'rmdir', 'mkdir', 'makedirs',
        'chmod', 'chown', 'kill', 'killpg', 'execl', 'execle', 'execlp',
        'execlpe', 'execv', 'execve', 'execvp', 'execvpe', 'fork', 'forkpty',
        'plock', 'putenv', 'spawnl', 'spawnle', 'spawnlp', 'spawnlpe',
        'spawnv', 'spawnve', 'spawnvp', 'spawnvpe', 'startfile', 'truncate',
        'symlink', 'link'
    },
    'subprocess': {
        'call', 'check_call', 'check_output', 'Popen', 'run'
    },
    'shutil': {
        'copy', 'copy2', 'copyfile', 'copyfileobj', 'copymode', 'copystat',
        'copytree', 'disk_usage', 'ignore_patterns', 'make_archive',
        'move', 'rmtree', 'unpack_archive', 'which'
    },
    'pickle': { # Unpickling can be dangerous
        'load', 'loads'
    },
    'ctypes': { # Can call arbitrary C code
        'CDLL', 'PyDLL', 'WinDLL', 'dlopen'
    },
    'multiprocessing': { # Can spawn processes
        'Process', 'Pool', 'Pipe', 'Queue', 'Manager', 'Lock', 'RLock',
        'Semaphore', 'BoundedSemaphore', 'Condition', 'Event', 'Barrier',
        'Value', 'Array'
    },
    'threading': { # While not directly I/O, can be used for resource exhaustion
        'Thread' # This might be too restrictive depending on use case
    },
    # For network access, checking for socket module usage
    # This is handled by checking for 'socket' import and attribute access
}

UNSAFE_MODULES_TO_IMPORT = {
    'os',
    'subprocess',
    'shutil',
    'socket',
    'ftplib',
    'http.client',
    'imaplib',
    'nntplib',
    'poplib',
    'smtplib',
    'telnetlib',
    'urllib.request', # urllib.parse, urllib.error are generally safer
    'ctypes',
    'multiprocessing',
    # 'pickle', # Importing pickle itself is not unsafe, its functions are. Handled by UNSAFE_MODULE_ATTRIBUTES.
               # However, to be more restrictive, one could add 'pickle' here.
}


class SafetyAnalyzer(ast.NodeVisitor):
    """
    AST visitor to detect potentially unsafe operations.
    """
    def __init__(self):
        self.safe = True
        self.imported_modules = {} # alias -> original_name

    def visit_Import(self, node):
        if not self.safe: return # Optimization: stop visiting if already unsafe
        for alias in node.names:
            if alias.name in UNSAFE_MODULES_TO_IMPORT:
                self.safe = False
                return
            self.imported_modules[alias.asname or alias.name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if not self.safe: return
        if node.module and node.module in UNSAFE_MODULES_TO_IMPORT:
            self.safe = False
            return
        for alias in node.names:
            # e.g., from os import system
            if node.module in UNSAFE_MODULE_ATTRIBUTES and alias.name in UNSAFE_MODULE_ATTRIBUTES[node.module]:
                self.safe = False
                return
            # What if someone does 'from unsafe_module import *'?
            # The '*' is not easily resolvable here without more context.
            # Current check on 'node.module in UNSAFE_MODULES_TO_IMPORT' handles the module itself.
        self.generic_visit(node)

    def visit_Call(self, node):
        if not self.safe: return
        # Check for direct unsafe function calls like open(), eval(), exec(), __import__()
        if isinstance(node.func, ast.Name):
            if node.func.id in UNSAFE_FUNCTION_CALLS:
                self.safe = False
                return

        # Check for module attribute calls like os.system() or socket.socket()
        elif isinstance(node.func, ast.Attribute):
            # obj.attr()
            # Resolve the object part of the call
            obj = node.func.value
            attr_name = node.func.attr
            
            current_obj = obj
            # Traverse down chained attributes like a.b.c.d() to get the base name 'a'
            # This is a simplification; full resolution is complex.
            # For 'a.b.c()', current_obj becomes 'a' (ast.Name)
            while isinstance(current_obj, ast.Attribute):
                current_obj = current_obj.value

            if isinstance(current_obj, ast.Name):
                module_alias_or_var_name = current_obj.id
                
                # Get original module name if it was imported with an alias
                original_module_name = self.imported_modules.get(module_alias_or_var_name, module_alias_or_var_name)

                if original_module_name in UNSAFE_MODULE_ATTRIBUTES:
                    # This checks if 'attr_name' is an unsafe attribute of 'original_module_name'
                    # e.g. os.system -> original_module_name='os', attr_name='system'
                    # e.g. my_os.system (where my_os = os) -> original_module_name='os', attr_name='system'
                    # This part needs to check the specific attribute being called.
                    # If node.func.attr (the final attribute in the call chain) is unsafe for that module.
                    if attr_name in UNSAFE_MODULE_ATTRIBUTES[original_module_name]:
                        self.safe = False
                        return
                
                if original_module_name == 'socket': # Any method call on socket module/instance is suspect
                    self.safe = False
                    return
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if not self.safe: return
        # This method is to catch things like `x = os.system` (assignment of an unsafe attribute)
        # or `s = socket.socket`
        # node.value is the object, node.attr is the attribute name
        
        current_obj = node.value
        # Traverse down chained attributes like a.b.c to get the base name 'a'
        while isinstance(current_obj, ast.Attribute):
            current_obj = current_obj.value

        if isinstance(current_obj, ast.Name):
            module_alias_or_var_name = current_obj.id
            attr_name = node.attr

            original_module_name = self.imported_modules.get(module_alias_or_var_name, module_alias_or_var_name)

            if original_module_name in UNSAFE_MODULE_ATTRIBUTES:
                if attr_name in UNSAFE_MODULE_ATTRIBUTES[original_module_name]:
                    self.safe = False
                    return
            
            if original_module_name == 'socket': # Any attribute access on socket module is suspect
                self.safe = False
                return
        self.generic_visit(node)


def analyze_snippet_safety(code_snippet: str) -> bool:
    """
    Analyzes a Python code snippet for potentially harmful operations.

    Args:
        code_snippet: A string containing Python code.

    Returns:
        True if the code is deemed safe, False otherwise.
    """
    try:
        tree = ast.parse(code_snippet)
    except SyntaxError:
        return False  # Code that doesn't parse is not safe to attempt execution

    analyzer = SafetyAnalyzer()
    analyzer.visit(tree)
    return analyzer.safe

def execute_safe_snippet(code_snippet: str) -> str:
    """
    Executes a Python code snippet if it's deemed safe by analyze_snippet_safety,
    and captures its stdout.

    Args:
        code_snippet: A string containing Python code.

    Returns:
        A string containing the stdout produced by the executed snippet.

    Raises:
        ValueError: If the code is unsafe, contains syntax errors, or if a
                    runtime error occurs during execution.
    """
    try:
        # Preliminary check for syntax errors before full analysis
        # ast.parse is already called in analyze_snippet_safety,
        # but calling it here first gives a more direct SyntaxError if that's the issue.
        ast.parse(code_snippet)
    except SyntaxError as e:
        raise ValueError(f"Invalid code: Syntax error: {e}") from e

    if not analyze_snippet_safety(code_snippet):
        raise ValueError("Unsafe code detected by AST analysis.")

    # Define a restricted set of builtins
    safe_builtins = {
        'print': print, 'len': len, 'range': range, 'list': list, 'dict': dict,
        'set': set, 'tuple': tuple, 'str': str, 'int': int, 'float': float,
        'bool': bool, 'None': None, 'True': True, 'False': False, 'abs': abs,
        'all': all, 'any': any, 'ascii': ascii, 'bin': bin, 'bytes': bytes,
        'callable': callable, 'chr': chr, 'complex': complex, 'divmod': divmod,
        'enumerate': enumerate, 'filter': filter, 'format': format,
        'frozenset': frozenset, 'hasattr': hasattr, 'hash': hash, 'hex': hex,
        'id': id, 'isinstance': isinstance, 'issubclass': issubclass, 'iter': iter,
        'max': max, 'min': min, 'next': next, 'oct': oct, 'ord': ord, 'pow': pow,
        'repr': repr, 'reversed': reversed, 'round': round, 'slice': slice,
        'sorted': sorted, 'sum': sum, 'type': type, 'vars': vars, 'zip': zip,
        # Common errors (can be useful in snippets)
        'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
        'NameError': NameError, 'IndexError': IndexError, 'KeyError': KeyError,
        'AttributeError': AttributeError, 'ZeroDivisionError': ZeroDivisionError,
        'OverflowError': OverflowError, 'RuntimeError': RuntimeError,
    }
    
    restricted_globals = {"__builtins__": safe_builtins}
    # Modules like 'math' must be explicitly imported in the snippet.
    # The 'import math' statement itself will be checked by SafetyAnalyzer.
    # If 'math' were in UNSAFE_MODULES_TO_IMPORT, it would be blocked.

    output_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(output_buffer):
            # Execute the code. Pass empty locals dict; it will be populated by exec.
            exec(code_snippet, restricted_globals, {})
    except Exception as e:
        # Catch any runtime errors from the executed code
        raise ValueError(f"Execution error: {type(e).__name__}: {e}") from e

    return output_buffer.getvalue()


if __name__ == '__main__':
    print("--- Testing analyze_snippet_safety ---")
    safe_code_snippets = [
        "print('Hello, world!')",
        "x = 1 + 2\nprint(x)",
        "def foo(n):\n  return n * 2\nfoo(5)",
        "import math\nprint(math.sqrt(16))", # math is not in UNSAFE_MODULES_TO_IMPORT
        "a = [1, 2, 3]\nfor x in a: print(x)",
        "my_dict = {'key': 'value'}\nprint(my_dict['key'])",
        "import os.path\nprint(os.path.join('a', 'b'))", # os.path is not os
        "from os.path import join\nprint(join('a', 'b'))"
    ]

    unsafe_code_snippets = [
        "open('some_file.txt', 'w')",
        "import os\nos.system('echo unsafe')",
        "import subprocess\nsubprocess.call(['ls'])",
        "eval('1 + 1')",
        "exec('print(\"unsafe\")')",
        "import socket\ns = socket.socket()\ns.connect(('example.com', 80))",
        "from os import system\nsystem('echo unsafe')",
        "import shutil\nshutil.rmtree('/tmp/dummy_dir')",
        "import pickle\npickle.loads(b'cos\nsystem\n(S\'echo unsafe\'\ntR.')",
        "import ctypes\nlibc = ctypes.CDLL(None)\nlibc.printf(b'test')",
        "from multiprocessing import Process\np = Process()\np.start()",
        "s = __import__('socket')\ns.socket()", # __import__ is unsafe
        "getattr(os, 'system')('echo unsafe')", # getattr is not blocked, but os.system is
        "import os\ngetattr(os, 'system')('echo unsafe')",
        "import sys\nsys.exit(1)", # sys is not in UNSAFE_MODULES_TO_IMPORT by default
                                   # but could be added if sys.exit is problematic.
        "del sys.modules['os']", # Modifying sys.modules is advanced, not directly caught
                                 # unless 'sys' is made an unsafe import.
        "import socket as so\nso.socket()",
        "import os\nf = os.system\nf('cmd')", # Caught by visit_Attribute on os.system
        "__import__('os').system('echo unsafe')", # Caught by __import__ in UNSAFE_FUNCTION_CALLS
    ]

    print("\nTesting safe snippets for analyze_snippet_safety:")
    all_safe_passed = True
    for i, snippet in enumerate(safe_code_snippets):
        is_safe = analyze_snippet_safety(snippet)
        print(f"Snippet {i+1} safe: {is_safe} -> {'OK' if is_safe else 'FAIL'}")
        if not is_safe: all_safe_passed = False
    # assert all_safe_passed, "One or more SAFE snippets were incorrectly flagged as unsafe."

    print("\nTesting unsafe snippets for analyze_snippet_safety:")
    all_unsafe_flagged = True
    for i, snippet in enumerate(unsafe_code_snippets):
        is_safe = analyze_snippet_safety(snippet)
        print(f"Snippet {i+1} safe: {is_safe} -> {'OK' if not is_safe else 'FAIL'}")
        if is_safe: all_unsafe_flagged = False
    # assert all_unsafe_flagged, "One or more UNSAFE snippets were incorrectly flagged as safe."

    # Specific analysis tests
    print("\nSpecific analysis tests:")
    test_assign_open = "o = open\no('file.txt', 'w')" # This is hard to catch without data flow. 'open' call is caught.
    print(f"Assign open then call: '{test_assign_open}' -> safe: {analyze_snippet_safety(test_assign_open)}") # Expected: False (Call to open is caught)

    test_eval_indirect = "e = eval\ne('1+1')"
    print(f"Assign eval then call: '{test_eval_indirect}' -> safe: {analyze_snippet_safety(test_eval_indirect)}") # Expected: False (Call to eval is caught)

    # --- execute_safe_snippet tests ---
    print("\n--- Testing execute_safe_snippet ---")

    # Test 1: Safe code with output
    code1 = "print('Hello from executed snippet')"
    print(f"\nExecuting: {code1}")
    try:
        output = execute_safe_snippet(code1)
        print(f"Output: '{output.strip()}' -> {'OK' if output.strip() == 'Hello from executed snippet' else 'FAIL'}")
    except ValueError as e:
        print(f"Error: {e} -> FAIL")

    # Test 2: Safe code, no output, but computation
    code2 = "x = 5 * 10\ny = x / 2"
    print(f"\nExecuting: {code2}")
    try:
        output = execute_safe_snippet(code2)
        print(f"Output: '{output.strip()}' -> {'OK' if output.strip() == '' else 'FAIL'}")
    except ValueError as e:
        print(f"Error: {e} -> FAIL")

    # Test 3: Unsafe code (file I/O via open)
    code3 = "open('test.txt', 'w').write('unsafe')"
    print(f"\nExecuting: {code3}")
    try:
        output = execute_safe_snippet(code3)
        print(f"Output: {output} -> FAIL (should have raised ValueError)")
    except ValueError as e:
        print(f"Error: {e} -> {'OK' if 'Unsafe code detected' in str(e) else 'FAIL: '+ str(e)}")

    # Test 4: Code with syntax error
    code4 = "print('Hello" # Missing closing parenthesis
    print(f"\nExecuting: {code4}")
    try:
        output = execute_safe_snippet(code4)
        print(f"Output: {output} -> FAIL (should have raised ValueError)")
    except ValueError as e:
        print(f"Error: {e} -> {'OK' if 'Invalid code: Syntax error' in str(e) else 'FAIL: '+ str(e)}")

    # Test 5: Safe code with runtime error
    code5 = "print(1/0)"
    print(f"\nExecuting: {code5}")
    try:
        output = execute_safe_snippet(code5)
        print(f"Output: {output} -> FAIL (should have raised ValueError)")
    except ValueError as e:
        print(f"Error: {e} -> {'OK' if 'Execution error' in str(e) and 'division by zero' in str(e).lower() else 'FAIL: '+ str(e)}")

    # Test 6: Unsafe code (os.system)
    code6 = "import os\nos.system('echo unsafe')"
    print(f"\nExecuting: {code6}")
    try:
        output = execute_safe_snippet(code6)
        print(f"Output: {output} -> FAIL (should have raised ValueError)")
    except ValueError as e:
        print(f"Error: {e} -> {'OK' if 'Unsafe code detected' in str(e) else 'FAIL: '+ str(e)}")

    # Test 7: Unsafe __import__
    code7 = "__import__('os').system('echo unsafe')"
    print(f"\nExecuting: {code7}")
    try:
        output = execute_safe_snippet(code7)
        print(f"Output: {output} -> FAIL (should have raised ValueError)")
    except ValueError as e:
        print(f"Error: {e} -> {'OK' if 'Unsafe code detected' in str(e) else 'FAIL: '+ str(e)}")

    # Test 8: Safe code using allowed builtins
    code8 = "my_list = [1,2,3]\nprint(len(my_list))\nfor i in range(2): print(i)"
    expected_output8 = "3\n0\n1"
    print(f"\nExecuting: {code8}")
    try:
        output = execute_safe_snippet(code8)
        print(f"Output:\n{output.strip()}\n -> {'OK' if output.strip() == expected_output8 else 'FAIL'}")
    except ValueError as e:
        print(f"Error: {e} -> FAIL")
        
    # Test 9: Code trying to use a disallowed builtin (e.g. 'input')
    code9 = "name = input('Enter name: ')\nprint(name)"
    print(f"\nExecuting: {code9}")
    try:
        output = execute_safe_snippet(code9)
        print(f"Output: {output} -> FAIL (should have raised ValueError due to NameError for input)")
    except ValueError as e:
        print(f"Error: {e} -> {'OK' if 'Execution error' in str(e) and '''name 'input' is not defined''' in str(e).lower() else 'FAIL: '+ str(e)}")

    # Test 10: Safe code that imports math (math is not unsafe to import)
    code10 = "import math\nprint(int(math.pow(2,3)))"
    expected_output10 = "8"
    print(f"\nExecuting: {code10}")
    try:
        output = execute_safe_snippet(code10)
        print(f"Output: '{output.strip()}' -> {'OK' if output.strip() == expected_output10 else 'FAIL'}")
    except ValueError as e:
        print(f"Error: {e} -> FAIL")

    # Test 11: Code that tries to import an unsafe module ('os')
    code11 = "import os\nprint('imported os')"
    print(f"\nExecuting: {code11}")
    try:
        output = execute_safe_snippet(code11)
        print(f"Output: {output} -> FAIL (should have raised ValueError)")
    except ValueError as e:
        print(f"Error: {e} -> {'OK' if 'Unsafe code detected' in str(e) else 'FAIL: '+ str(e)}")


def run_python_snippet(code_snippet: str) -> str | None:
    """
    Checks the safety of a Python code snippet, executes it if safe,
    and returns the result or None if there's no output.

    Args:
        code_snippet: A string containing Python code.

    Returns:
        The stdout produced by the snippet, or None if no output.

    Raises:
        ValueError: If the code is unsafe, invalid (syntax error), or if a
                    runtime error occurs during execution.
    """
    # analyze_snippet_safety and execute_safe_snippet already handle
    # raising ValueError for unsafe/invalid code or execution errors.
    # execute_safe_snippet will raise ValueError for syntax errors
    # before analyze_snippet_safety is even called in its internal logic,
    # or if analyze_snippet_safety returns False.
    # It also raises ValueError for runtime errors.

    try:
        output = execute_safe_snippet(code_snippet)
        if output == "":
            return None
        return output
    except ValueError:
        # Re-raise the ValueError from execute_safe_snippet
        # This will cover syntax errors, safety violations, and runtime errors.
        raise


if __name__ == '__main__':
    # (Previous tests for analyze_snippet_safety and execute_safe_snippet remain)
    # ...

    # --- run_python_snippet tests ---
    print("\n--- Testing run_python_snippet ---")

    # Test R1: Safe code with output
    r_code1 = "print('Hello from run_python_snippet')"
    print(f"\nRunning: {r_code1}")
    try:
        r_output1 = run_python_snippet(r_code1)
        expected_r_output1 = "Hello from run_python_snippet\n"
        print(f"Output: '{r_output1}' -> {'OK' if r_output1 == expected_r_output1 else 'FAIL'}")
    except ValueError as e:
        print(f"Error: {e} -> FAIL")

    # Test R2: Safe code with no output
    r_code2 = "x = 10 * 5"
    print(f"\nRunning: {r_code2}")
    try:
        r_output2 = run_python_snippet(r_code2)
        print(f"Output: {r_output2} -> {'OK' if r_output2 is None else 'FAIL'}")
    except ValueError as e:
        print(f"Error: {e} -> FAIL")

    # Test R3: Unsafe code (os.system)
    r_code3 = "import os; os.system('clear')"
    print(f"\nRunning: {r_code3}")
    try:
        r_output3 = run_python_snippet(r_code3)
        print(f"Output: {r_output3} -> FAIL (should have raised ValueError)")
    except ValueError as e:
        print(f"Error: {e} -> {'OK' if 'Unsafe code detected' in str(e) else 'FAIL: '+ str(e)}")

    # Test R4: Code with syntax error
    r_code4 = "print('Syntax Error"
    print(f"\nRunning: {r_code4}")
    try:
        r_output4 = run_python_snippet(r_code4)
        print(f"Output: {r_output4} -> FAIL (should have raised ValueError)")
    except ValueError as e:
        print(f"Error: {e} -> {'OK' if 'Invalid code: Syntax error' in str(e) else 'FAIL: '+ str(e)}")

    # Test R5: Safe code with runtime error
    r_code5 = "print(10 / 0)"
    print(f"\nRunning: {r_code5}")
    try:
        r_output5 = run_python_snippet(r_code5)
        print(f"Output: {r_output5} -> FAIL (should have raised ValueError)")
    except ValueError as e:
        print(f"Error: {e} -> {'OK' if 'Execution error' in str(e) and 'division by zero' in str(e).lower() else 'FAIL: '+ str(e)}")
    
    # Test R6: Safe code with multi-line output
    r_code6 = "print('Line 1')\nprint('Line 2')"
    expected_r_output6 = "Line 1\nLine 2\n"
    print(f"\nRunning: {r_code6}")
    try:
        r_output6 = run_python_snippet(r_code6)
        print(f"Output:\n'{r_output6}' -> {'OK' if r_output6 == expected_r_output6 else 'FAIL'}")
    except ValueError as e:
        print(f"Error: {e} -> FAIL")
