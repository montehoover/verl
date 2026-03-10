import ast
import io
import contextlib

# Define a list of allowed built-in functions for the restricted execution scope
ALLOWED_BUILTINS = {
    "print": print,
    "abs": abs,
    "all": all,
    "any": any,
    "ascii": ascii,
    "bin": bin,
    "bool": bool,
    "bytearray": bytearray,
    "bytes": bytes,
    "callable": callable,
    "chr": chr,
    "complex": complex,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "hasattr": hasattr,
    "hash": hash,
    "hex": hex,
    "id": id,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
}

def analyze_ast_for_safety(code_string: str) -> bool:
    """
    Parses a string of Python code into an AST and checks for
    potentially harmful operations.

    Args:
        code_string: The Python code to analyze.

    Returns:
        True if the script is deemed safe, False otherwise.
    """
    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        # If the code can't be parsed, it can't be run,
        # but we'll consider it "unsafe" from an execution standpoint.
        return False

    for node in ast.walk(tree):
        # Disallow imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False
        
        # Disallow calls to certain built-in functions
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ['open', 'eval', 'exec']:
                    return False
            # Disallow attribute calls like os.system
            if isinstance(node.func, ast.Attribute):
                # This is a simple check, could be more sophisticated
                # e.g., checking if 'os' or 'subprocess' was imported and then used.
                # For now, let's be very restrictive.
                # A more robust check would track imported modules.
                if node.func.attr in ['system', 'popen', 'spawn', 'call', 'run']:
                    return False

    return True

def execute_safe_script(code_string: str) -> str:
    """
    Executes a Python script if it's deemed safe by AST analysis,
    capturing and returning its stdout.

    Args:
        code_string: The Python code to execute.

    Returns:
        The captured stdout from the script execution.

    Raises:
        ValueError: If the script contains prohibited operations.
        Any other exception raised during script execution will propagate.
    """
    if not analyze_ast_for_safety(code_string):
        raise ValueError("Script contains prohibited operations and cannot be executed.")

    # Prepare a restricted global scope
    safe_globals = {
        "__builtins__": ALLOWED_BUILTINS
    }
    
    # Capture stdout
    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code_string, safe_globals, {}) # Use an empty dict for locals
        return stdout_capture.getvalue()
    finally:
        stdout_capture.close()

def safe_run_script(code_string: str) -> str | None:
    """
    Analyzes a Python script for safety and executes it if deemed safe.

    Args:
        code_string: The Python code to analyze and execute.

    Returns:
        The captured stdout from the script execution as a string.
        Returns None if the script executes successfully but produces no stdout.

    Raises:
        ValueError: If the script is deemed unsafe, contains prohibited
                    operations, or has invalid syntax.
        Any other exception raised during script execution will propagate.
    """
    try:
        # execute_safe_script already calls analyze_ast_for_safety
        # and will raise ValueError if the script is unsafe or has syntax errors.
        output = execute_safe_script(code_string)
        return output if output else None
    except ValueError:  # Re-raise ValueError from execute_safe_script
        raise
    except Exception:  # Propagate other execution errors
        raise

if __name__ == '__main__':
    safe_code_1 = "print('Hello, world!')"
    safe_code_2 = "x = 1 + 2\nprint(x)"
    safe_code_3 = """
def greet(name):
    return f"Hello, {name}"
print(greet("User"))
"""

    unsafe_code_1 = "import os\nos.system('echo unsafe')"
    unsafe_code_2 = "open('some_file.txt', 'w').write('dangerous')"
    unsafe_code_3 = "eval('1 + 1')" # eval itself is disallowed
    unsafe_code_4 = "exec('print(\"executed\")')" # exec itself is disallowed
    unsafe_code_5 = "__import__('os').system('echo unsafe')" # using __import__
    unsafe_code_6 = "import subprocess\nsubprocess.call(['ls'])"
    unsafe_code_7 = """
import os
def do_something():
    os.system("echo 'dangerous'")
do_something()
"""

    print(f"'{safe_code_1[:20]}...' is safe: {analyze_ast_for_safety(safe_code_1)}")
    print(f"'{safe_code_2[:20]}...' is safe: {analyze_ast_for_safety(safe_code_2)}")
    print(f"'{safe_code_3[:20]}...' is safe: {analyze_ast_for_safety(safe_code_3)}")

    print(f"'{unsafe_code_1[:20]}...' is safe: {analyze_ast_for_safety(unsafe_code_1)}")
    print(f"'{unsafe_code_2[:20]}...' is safe: {analyze_ast_for_safety(unsafe_code_2)}")
    print(f"'{unsafe_code_3[:20]}...' is safe: {analyze_ast_for_safety(unsafe_code_3)}")
    print(f"'{unsafe_code_4[:20]}...' is safe: {analyze_ast_for_safety(unsafe_code_4)}")
    print(f"'{unsafe_code_5[:20]}...' is safe: {analyze_ast_for_safety(unsafe_code_5)}")
    print(f"'{unsafe_code_6[:20]}...' is safe: {analyze_ast_for_safety(unsafe_code_6)}")
    print(f"'{unsafe_code_7[:20]}...' is safe: {analyze_ast_for_safety(unsafe_code_7)}")

    # Test syntax error
    syntax_error_code = "print('Hello'"
    print(f"'{syntax_error_code[:20]}...' is safe: {analyze_ast_for_safety(syntax_error_code)}")

    print("\n--- Testing execute_safe_script ---")
    
    # Test 1: Safe code that prints
    code_to_exec_1 = "print('Hello from executed script!')\nx = 10\nprint(f'x is {x}')"
    print(f"Executing: {code_to_exec_1[:30]}...")
    try:
        output = execute_safe_script(code_to_exec_1)
        print(f"Output:\n{output.strip()}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: Safe code, no print (result should be empty string)
    code_to_exec_2 = "y = 5 * 5"
    print(f"\nExecuting: {code_to_exec_2[:30]}...")
    try:
        output = execute_safe_script(code_to_exec_2)
        print(f"Output:\n'{output.strip()}'") # Expecting empty string
    except Exception as e:
        print(f"Error: {e}")

    # Test 3: Unsafe code (import)
    code_to_exec_3 = "import os\nprint(os.getcwd())"
    print(f"\nExecuting: {code_to_exec_3[:30]}...")
    try:
        output = execute_safe_script(code_to_exec_3)
        print(f"Output:\n{output.strip()}")
    except ValueError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    # Test 4: Unsafe code (open)
    code_to_exec_4 = "with open('test.txt', 'w') as f: f.write('hello')" # AST check disallows open
    print(f"\nExecuting: {code_to_exec_4[:30]}...")
    try:
        output = execute_safe_script(code_to_exec_4)
        print(f"Output:\n{output.strip()}")
    except ValueError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    # Test 5: AST-safe code with runtime error
    code_to_exec_5 = "print('About to divide by zero...')\nprint(1/0)"
    print(f"\nExecuting: {code_to_exec_5[:30]}...")
    try:
        output = execute_safe_script(code_to_exec_5)
        print(f"Output:\n{output.strip()}")
    except ZeroDivisionError as e:
        print(f"Caught expected runtime error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # Test 6: Code trying to use a disallowed builtin (eval) - caught by AST check
    code_to_exec_6 = "eval('1+1')"
    print(f"\nExecuting: {code_to_exec_6[:30]}...")
    try:
        output = execute_safe_script(code_to_exec_6)
        print(f"Output:\n{output.strip()}")
    except ValueError as e: 
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    # Test 7: Code trying to use a builtin not in ALLOWED_BUILTINS (e.g. input)
    # 'input' is not in ALLOWED_BUILTINS and not explicitly in the AST check's dangerous list.
    # This should result in a NameError during exec.
    code_to_exec_7 = "name = input('Enter name: ')\nprint(name)"
    print(f"\nExecuting: {code_to_exec_7[:30]}...")
    try:
        output = execute_safe_script(code_to_exec_7)
        print(f"Output:\n{output.strip()}")
    except NameError as e:
        print(f"Caught expected runtime error (NameError for input): {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # Test 8: Code that is syntactically incorrect
    code_to_exec_8 = "print('Hello'"
    print(f"\nExecuting syntactically incorrect code: {code_to_exec_8[:30]}...")
    try:
        # analyze_ast_for_safety will return False for syntax errors,
        # leading to ValueError from execute_safe_script
        output = execute_safe_script(code_to_exec_8)
        print(f"Output:\n{output.strip()}")
    except ValueError as e:
        print(f"Caught expected error for syntax issue: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    print("\n--- Testing safe_run_script ---")

    # Test 1: Safe script with output
    srs_code_1 = "print('Hello from safe_run_script!')"
    print(f"\nExecuting with safe_run_script: {srs_code_1[:30]}...")
    try:
        result = safe_run_script(srs_code_1)
        print(f"Result type: {type(result)}, Result: '{result}'")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: Safe script with no output (should return None)
    srs_code_2 = "a = 10 + 5"
    print(f"\nExecuting with safe_run_script: {srs_code_2[:30]}...")
    try:
        result = safe_run_script(srs_code_2)
        print(f"Result type: {type(result)}, Result: {result}") # Expect None
    except Exception as e:
        print(f"Error: {e}")

    # Test 3: Unsafe script (import os)
    srs_code_3 = "import os\nprint('unsafe')"
    print(f"\nExecuting with safe_run_script: {srs_code_3[:30]}...")
    try:
        result = safe_run_script(srs_code_3)
        print(f"Result type: {type(result)}, Result: {result}")
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    # Test 4: Script with invalid syntax
    srs_code_4 = "print('missing parenthesis"
    print(f"\nExecuting with safe_run_script: {srs_code_4[:30]}...")
    try:
        result = safe_run_script(srs_code_4)
        print(f"Result type: {type(result)}, Result: {result}")
    except ValueError as e:
        print(f"Caught expected ValueError: {e}") # analyze_ast_for_safety -> False -> ValueError in execute_safe_script
    except Exception as e:
        print(f"Unexpected error: {e}")

    # Test 5: Script with runtime error (ZeroDivisionError)
    srs_code_5 = "x = 1 / 0"
    print(f"\nExecuting with safe_run_script: {srs_code_5[:30]}...")
    try:
        result = safe_run_script(srs_code_5)
        print(f"Result type: {type(result)}, Result: {result}")
    except ZeroDivisionError as e:
        print(f"Caught expected ZeroDivisionError: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # Test 6: Script trying to use disallowed builtin (eval)
    srs_code_6 = "eval('1+1')"
    print(f"\nExecuting with safe_run_script: {srs_code_6[:30]}...")
    try:
        result = safe_run_script(srs_code_6)
        print(f"Result type: {type(result)}, Result: {result}")
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
