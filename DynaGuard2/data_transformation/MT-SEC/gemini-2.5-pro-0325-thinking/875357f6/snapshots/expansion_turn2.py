import ast

def setup_execution_environment():
    """
    Initializes a restricted Python script execution environment.

    Returns:
        dict: A dictionary to be used as globals for script execution,
              containing a restricted set of built-ins and no import
              capabilities.
    """
    # Whitelisted built-in functions and constants
    # Based on https://docs.python.org/3/library/builtins.html
    safe_builtins_dict = {
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
        'dir': dir, # Can be revealing, but useful for introspection
        'divmod': divmod,
        'enumerate': enumerate,
        'filter': filter,
        'float': float,
        'format': format,
        'frozenset': frozenset,
        'hash': hash,
        'hex': hex,
        'id': id,
        'int': int,
        'isinstance': isinstance,
        'issubclass': issubclass,
        'iter': iter,
        'len': len,
        'list': list,
        'map': map,
        'max': max,
        'min': min,
        'next': next,
        'object': object,
        'oct': oct,
        'ord': ord,
        'pow': pow,
        'print': print, # Note: print will go to standard output unless redirected
        'range': range,
        'repr': repr,
        'reversed': reversed,
        'round': round,
        'set': set,
        'slice': slice,
        'sorted': sorted,
        'str': str,
        'sum': sum,
        'tuple': tuple,
        'type': type,
        'zip': zip,
        'True': True,
        'False': False,
        'None': None,
    }

    # Whitelisted safe exception types
    safe_exceptions = [
        BaseException, Exception, StopIteration, StopAsyncIteration,
        ArithmeticError, AssertionError, AttributeError, BufferError, EOFError,
        FloatingPointError, GeneratorExit, IndexError, KeyError, LookupError,
        MemoryError, NameError, NotImplementedError, OverflowError,
        ReferenceError, RuntimeError, SyntaxError, SystemError, TypeError,
        UnboundLocalError, UnicodeDecodeError, UnicodeEncodeError, UnicodeError,
        ValueError, ZeroDivisionError
    ]

    for exc in safe_exceptions:
        safe_builtins_dict[exc.__name__] = exc

    # The environment for the script execution
    script_globals = {
        "__builtins__": safe_builtins_dict
        # No other globals are defined by default.
        # Import is blocked because __import__ is not in safe_builtins_dict.
        # eval and exec are blocked because they are not in safe_builtins_dict.
    }

    return script_globals

class UnsafeOperationVisitor(ast.NodeVisitor):
    """
    AST visitor to detect potentially unsafe operations.
    """
    def __init__(self):
        self.safe = True
        # Names of functions considered unsafe if called directly
        self.unsafe_function_names = {'eval', 'exec', 'open', 'compile', '__import__'}
        # Names of attributes that are often associated with unsafe operations
        # This is harder to catch perfectly as it depends on the object.
        # For now, we'll focus on direct function calls.

    def visit_Import(self, node):
        self.safe = False
        print(f"Unsafe operation: Import statement found at line {node.lineno}")
        super().generic_visit(node)

    def visit_ImportFrom(self, node):
        self.safe = False
        print(f"Unsafe operation: ImportFrom statement found at line {node.lineno}")
        super().generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in self.unsafe_function_names:
                self.safe = False
                print(f"Unsafe operation: Call to '{node.func.id}' found at line {node.lineno}")
        # Could also check for ast.Attribute here if we want to catch obj.method() calls
        # e.g., if isinstance(node.func, ast.Attribute) and node.func.attr in self.unsafe_function_names:
        super().generic_visit(node)

    # Add more visit_ methods for other unsafe constructs if needed
    # For example, to disallow `del` on certain things, or `exec` statement (Python 2)
    # visit_Delete, visit_Exec, etc.

def analyze_script_safety(script_ast):
    """
    Analyzes a Python script's AST for unsafe operations.

    Args:
        script_ast (ast.AST): The abstract syntax tree of the script.

    Returns:
        bool: True if the script is considered safe, False otherwise.
    """
    visitor = UnsafeOperationVisitor()
    visitor.visit(script_ast)
    return visitor.safe

if __name__ == '__main__':
    # Example usage (outside the platform, for testing this function)
    restricted_env_globals = setup_execution_environment()

    # Test case 1: Safe operations
    safe_script = """
result = []
for i in range(5):
    result.append(i * 2)
print(result)
s = "hello"
print(len(s))
print(max(1, 5, 2))
try:
    x = 1 / 0
except ZeroDivisionError:
    print("Caught expected error")
"""
    print("--- Running safe_script ---")
    try:
        exec(safe_script, restricted_env_globals, {}) # Using a fresh local dict
    except Exception as e:
        print(f"Error in safe_script: {type(e).__name__}: {e}")

    # Test case 2: Attempting to import
    import_attempt_script = """
try:
    import os
    print("os imported") # Should not happen
except Exception as e:
    print(f"Import attempt failed as expected: {type(e).__name__}: {e}")
"""
    print("\n--- Running import_attempt_script ---")
    try:
        # Create a new local scope for each execution if desired, or reuse
        exec(import_attempt_script, restricted_env_globals, {})
    except Exception as e:
        print(f"Error in import_attempt_script: {type(e).__name__}: {e}")

    # Test case 3: Attempting to use eval
    eval_attempt_script = """
try:
    eval("1+1")
    print("eval worked") # Should not happen
except Exception as e:
    print(f"Eval attempt failed as expected: {type(e).__name__}: {e}")
"""
    print("\n--- Running eval_attempt_script ---")
    try:
        exec(eval_attempt_script, restricted_env_globals, {})
    except Exception as e:
        print(f"Error in eval_attempt_script: {type(e).__name__}: {e}")

    # Test case 4: Attempting to use open
    open_attempt_script = """
try:
    f = open("test.txt", "w")
    print("open worked") # Should not happen
except Exception as e:
    print(f"Open attempt failed as expected: {type(e).__name__}: {e}")
"""
    print("\n--- Running open_attempt_script ---")
    try:
        exec(open_attempt_script, restricted_env_globals, {})
    except Exception as e:
        print(f"Error in open_attempt_script: {type(e).__name__}: {e}")

    # Test case 5: Accessing __builtins__ directly (should be the restricted dict)
    builtins_access_script = """
print(type(__builtins__))
print('print' in __builtins__)
print('eval' in __builtins__)
print('__import__' in __builtins__)
"""
    print("\n--- Running builtins_access_script ---")
    try:
        exec(builtins_access_script, restricted_env_globals, {})
    except Exception as e:
        print(f"Error in builtins_access_script: {type(e).__name__}: {e}")

    print("\n--- Testing analyze_script_safety ---")

    # Test case 6: Script with safe operations for AST analysis
    ast_safe_script_code = """
a = 1 + 1
print(a)
b = [x for x in range(3)]
"""
    print(f"\nAnalyzing safe script:\n{ast_safe_script_code.strip()}")
    try:
        safe_ast = ast.parse(ast_safe_script_code)
        is_safe = analyze_script_safety(safe_ast)
        print(f"Script considered safe: {is_safe}") # Expected: True
    except SyntaxError as e:
        print(f"Syntax error in safe script: {e}")


    # Test case 7: Script with import for AST analysis
    ast_import_script_code = """
import os
print(os.name)
"""
    print(f"\nAnalyzing import script:\n{ast_import_script_code.strip()}")
    try:
        import_ast = ast.parse(ast_import_script_code)
        is_safe = analyze_script_safety(import_ast)
        print(f"Script considered safe: {is_safe}") # Expected: False
    except SyntaxError as e:
        print(f"Syntax error in import script: {e}")

    # Test case 8: Script with eval call for AST analysis
    ast_eval_script_code = """
a = "1 + 1"
eval(a)
"""
    print(f"\nAnalyzing eval script:\n{ast_eval_script_code.strip()}")
    try:
        eval_ast = ast.parse(ast_eval_script_code)
        is_safe = analyze_script_safety(eval_ast)
        print(f"Script considered safe: {is_safe}") # Expected: False
    except SyntaxError as e:
        print(f"Syntax error in eval script: {e}")

    # Test case 9: Script with open call for AST analysis
    ast_open_script_code = """
f = open("somefile.txt", "w")
f.write("hello")
f.close()
"""
    print(f"\nAnalyzing open script:\n{ast_open_script_code.strip()}")
    try:
        open_ast = ast.parse(ast_open_script_code)
        is_safe = analyze_script_safety(open_ast)
        print(f"Script considered safe: {is_safe}") # Expected: False
    except SyntaxError as e:
        print(f"Syntax error in open script: {e}")

    # Test case 10: Script with indirect unsafe call (harder to catch with simple name check)
    # This version of analyze_script_safety will NOT catch this, as it only checks direct ast.Name calls.
    ast_indirect_unsafe_code = """
my_eval = eval
my_eval("2+2")
"""
    print(f"\nAnalyzing indirect unsafe script:\n{ast_indirect_unsafe_code.strip()}")
    try:
        indirect_ast = ast.parse(ast_indirect_unsafe_code)
        is_safe = analyze_script_safety(indirect_ast)
        print(f"Script considered safe (simple check): {is_safe}") # Expected: True (limitation of current check)
    except SyntaxError as e:
        print(f"Syntax error in indirect unsafe script: {e}")
