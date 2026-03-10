import ast

# Define a dictionary of safe built-in functions and constants
SAFE_BUILTINS = {
    'abs': abs, 'all': all, 'any': any, 'ascii': ascii, 'bin': bin, 'bool': bool,
    'bytearray': bytearray, 'bytes': bytes, 'callable': callable, 'chr': chr,
    'complex': complex, 'dict': dict, 'divmod': divmod, 'enumerate': enumerate,
    'filter': filter, 'float': float, 'format': format, 'frozenset': frozenset,
    # 'getattr': getattr, # Potentially risky, consider if truly needed and its implications
    'hasattr': hasattr, 'hash': hash, 'hex': hex, 'id': id, 'int': int,
    'isinstance': isinstance, 'issubclass': issubclass, 'iter': iter, 'len': len,
    'list': list, 'map': map, 'max': max, 'min': min, 'next': next, 'oct': oct,
    'ord': ord, 'pow': pow, 'print': print, 'range': range, 'repr': repr,
    'reversed': reversed, 'round': round, 'set': set, 'slice': slice,
    'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple, 'type': type,
    'zip': zip,
    # Commonly allowed exceptions
    'ArithmeticError': ArithmeticError, 'AssertionError': AssertionError,
    'AttributeError': AttributeError, 'EOFError': EOFError, 'Exception': Exception,
    'FloatingPointError': FloatingPointError, 'GeneratorExit': GeneratorExit,
    'ImportError': ImportError, 'IndexError': IndexError, 'KeyError': KeyError,
    'LookupError': LookupError, 'MemoryError': MemoryError, 'NameError': NameError,
    'NotImplementedError': NotImplementedError, 'OSError': OSError,
    'OverflowError': OverflowError, 'ReferenceError': ReferenceError,
    'RuntimeError': RuntimeError, 'StopIteration': StopIteration,
    'SyntaxError': SyntaxError, 'SystemError': SystemError, 'TypeError': TypeError,
    'ValueError': ValueError, 'ZeroDivisionError': ZeroDivisionError,
    # Constants
    'True': True, 'False': False, 'None': None,
    '...': Ellipsis,
    'NotImplemented': NotImplemented,
    # It's crucial to NOT include:
    # eval, exec, open, __import__, compile, dir, globals, locals, input,
    # memoryview, object, property, setattr, delattr, file, socket, etc.
}

# The global scope for exec/eval, restricting access to builtins
restricted_globals = {"__builtins__": SAFE_BUILTINS}
# To allow specific safe modules, you would import them and add to restricted_globals:
# import math
# restricted_globals['math'] = math


def safe_run_script(script_code: str):
    """
    Executes a user-provided Python script in a restricted environment.

    Args:
        script_code: The string containing the Python script.

    Returns:
        The result of the last evaluated expression in the script, or None if
        the script is empty, the last line is not an expression, or if the
        script only contains statements.

    Raises:
        ValueError: If the script involves prohibited operations (e.g., file access,
                    network requests, disallowed builtins) or contains invalid syntax.
    """
    local_namespace = {}
    try:
        # Ensure the script is not empty or just whitespace
        if not script_code.strip():
            return None

        # Parse the script into an Abstract Syntax Tree (AST)
        # This can raise SyntaxError, which we'll catch below
        parsed_ast = ast.parse(script_code, mode='exec')

        if not parsed_ast.body:  # Handles scripts with only comments or whitespace
            return None

        # Check for prohibited operations by walking the AST
        # For simplicity, this example relies on the restricted_globals for runtime checks.
        # A more robust solution would involve an AST visitor to disallow certain nodes
        # (e.g., ast.Import, ast.ImportFrom, ast.Call to 'open', etc.) before execution.
        # However, relying on restricted_globals already prevents access to 'open', '__import__', etc.

        last_node = parsed_ast.body[-1]

        if isinstance(last_node, ast.Expr):
            # If there are statements before the final expression, execute them first.
            if len(parsed_ast.body) > 1:
                exec_module = ast.Module(body=parsed_ast.body[:-1], type_ignores=[])
                code_obj_exec = compile(exec_module, '<string>', 'exec')
                exec(code_obj_exec, restricted_globals, local_namespace)

            # Compile and evaluate the last expression statement
            eval_expression_node = ast.Expression(body=last_node.value)
            code_obj_eval = compile(eval_expression_node, '<string>', 'eval')
            result = eval(code_obj_eval, restricted_globals, local_namespace)
            return result
        else:
            # The last statement is not an expression. Execute the entire script.
            code_obj_exec = compile(parsed_ast, '<string>', 'exec')
            exec(code_obj_exec, restricted_globals, local_namespace)
            return None  # exec returns None, and the last line wasn't an expression.

    except SyntaxError as e:
        raise ValueError(f"Invalid syntax in script: {e}") from e
    except NameError as e:
        # This often indicates use of a disallowed function/variable (e.g., 'open', 'os')
        raise ValueError(f"Prohibited operation or undefined name: {e}") from e
    except Exception as e:
        # Catch other potential runtime errors and wrap them in ValueError
        # to signal a problem with the script's execution in the safe environment.
        raise ValueError(f"Error during script execution: {e}") from e


if __name__ == '__main__':
    # Example usage:
    script1 = "a = 10\nb = 20\na + b"
    print(f"Script 1:\n{script1}\nResult: {safe_run_script(script1)}\n")

    script2 = "x = 5\ny = 3\nx * y"
    print(f"Script 2:\n{script2}\nResult: {safe_run_script(script2)}\n")

    script3 = "z = 'hello'\nlen(z.upper())"
    print(f"Script 3:\n{script3}\nResult: {safe_run_script(script3)}\n")

    script4 = "k = 100  # Ends with assignment"
    print(f"Script 4:\n{script4}\nResult: {safe_run_script(script4)}\n")

    script5 = "print('Hello from script')\n7 * 7" # print is allowed
    print(f"Script 5 (output will appear above result line):\n{script5}")
    print(f"Result: {safe_run_script(script5)}\n")

    script_empty = ""
    print(f"Script Empty:\n'{script_empty}'\nResult: {safe_run_script(script_empty)}\n")

    script_only_comment = "# This is a comment\n# and another one"
    print(f"Script Only Comment:\n'{script_only_comment}'\nResult: {safe_run_script(script_only_comment)}\n")

    script_just_expression = "99 + 1"
    print(f"Script Just Expression:\n'{script_just_expression}'\nResult: {safe_run_script(script_just_expression)}\n")

    # Error handling examples (expecting ValueError)
    script_syntax_error = "a = 1 +"
    print(f"Script Syntax Error:\n{script_syntax_error}")
    try:
        safe_run_script(script_syntax_error)
    except ValueError as e:
        print(f"Caught expected ValueError: {e}\n")

    script_runtime_type_error = "'text' + 123" # TypeError
    print(f"Script Runtime Error (TypeError):\n{script_runtime_type_error}")
    try:
        safe_run_script(script_runtime_type_error)
    except ValueError as e:
        print(f"Caught expected ValueError: {e}\n")

    script_zero_division = "b = 0\n10 / b"
    print(f"Script Runtime Error (ZeroDivisionError):\n{script_zero_division}")
    try:
        safe_run_script(script_zero_division)
    except ValueError as e:
        print(f"Caught expected ValueError: {e}\n")

    # Security tests (these should raise ValueError due to NameError for disallowed functions)
    script_file_access = "open('test.txt', 'w')"
    print(f"Script File Access (should fail with ValueError):\n{script_file_access}")
    try:
        safe_run_script(script_file_access)
    except ValueError as e:
        print(f"Caught expected ValueError (for 'open'): {e}\n")
    except Exception as e:
        print(f"Caught UNEXPECTED error for file access: {e}\n")

    script_import_os = "import os\nos.getcwd()"
    print(f"Script Import OS (should fail with ValueError):\n{script_import_os}")
    try:
        # 'import' itself relies on '__import__', which is not in SAFE_BUILTINS
        safe_run_script(script_import_os)
    except ValueError as e:
        print(f"Caught expected ValueError (for '__import__'): {e}\n")
    except Exception as e:
        print(f"Caught UNEXPECTED error for import os: {e}\n")

    script_eval_inside = "eval('1+1')"
    print(f"Script eval inside (should fail with ValueError):\n{script_eval_inside}")
    try:
        safe_run_script(script_eval_inside)
    except ValueError as e:
        print(f"Caught expected ValueError (for 'eval'): {e}\n")
    except Exception as e:
        print(f"Caught UNEXPECTED error for eval inside: {e}\n")

    script_exec_inside = "exec('a=1')"
    print(f"Script exec inside (should fail with ValueError):\n{script_exec_inside}")
    try:
        safe_run_script(script_exec_inside)
    except ValueError as e:
        print(f"Caught expected ValueError (for 'exec'): {e}\n")
    except Exception as e:
        print(f"Caught UNEXPECTED error for exec inside: {e}\n")

    script_dangerous_getattr = "x = ().__class__.__bases__[0].__subclasses__()[0].__init__.__globals__['sys']"
    print(f"Script dangerous getattr-like (should fail with ValueError):\n{script_dangerous_getattr}")
    try:
        # This relies on 'sys' not being in restricted_globals.
        # If 'type' is allowed, parts of the chain might work, but the final access to 'sys' should fail.
        safe_run_script(script_dangerous_getattr)
    except ValueError as e:
        print(f"Caught expected ValueError (likely for 'sys' or due to restricted attribute access): {e}\n")
    except Exception as e:
        print(f"Caught UNEXPECTED error for dangerous getattr: {e}\n")
