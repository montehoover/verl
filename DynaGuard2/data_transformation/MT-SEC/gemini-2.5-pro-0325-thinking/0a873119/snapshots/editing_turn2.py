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


def execute_script(script_text: str):
    """
    Executes a multi-line Python script in a restricted environment and
    returns the result of the last expression statement.

    Args:
        script_text: The string containing the Python script.

    Returns:
        The result of the last evaluated expression in the script, or None if
        the script is empty or the last line is not an expression.
    """
    local_namespace = {}
    try:
        # Ensure the script is not empty or just whitespace
        if not script_text.strip():
            return None

        # Parse the script into an Abstract Syntax Tree (AST)
        parsed_ast = ast.parse(script_text, mode='exec')

        if not parsed_ast.body:  # Handles scripts with only comments or whitespace
            return None

        # If the last statement in the script is an expression, we aim to return its value.
        # Otherwise, we execute the script and return None.
        last_node = parsed_ast.body[-1]

        if isinstance(last_node, ast.Expr):
            # If there are statements before the final expression, execute them first.
            if len(parsed_ast.body) > 1:
                # Compile all statements except the last one
                exec_module = ast.Module(body=parsed_ast.body[:-1], type_ignores=[])
                code_obj_exec = compile(exec_module, '<string>', 'exec')
                exec(code_obj_exec, restricted_globals, local_namespace)

            # Compile and evaluate the last expression statement
            # ast.Expression is used for eval mode with an ast.Expr node
            eval_expression_node = ast.Expression(body=last_node.value)
            code_obj_eval = compile(eval_expression_node, '<string>', 'eval')
            result = eval(code_obj_eval, restricted_globals, local_namespace)
            return result
        else:
            # The last statement is not an expression (e.g., assignment, def, class).
            # Execute the entire script.
            code_obj_exec = compile(parsed_ast, '<string>', 'exec')
            exec(code_obj_exec, restricted_globals, local_namespace)
            return None  # exec returns None, and the last line wasn't an expression.

    except SyntaxError as e:
        print(f"Syntax error in script: {e}")
        raise
    except Exception as e:
        # Catching other exceptions that might occur during compilation or execution
        print(f"Error executing script '{script_text[:50]}...': {e}")
        raise


if __name__ == '__main__':
    # Example usage:
    script1 = "a = 10\nb = 20\na + b"
    print(f"Script 1:\n{script1}\nResult: {execute_script(script1)}\n")

    script2 = "x = 5\ny = 3\nx * y"
    print(f"Script 2:\n{script2}\nResult: {execute_script(script2)}\n")

    script3 = "z = 'hello'\nlen(z.upper())"
    print(f"Script 3:\n{script3}\nResult: {execute_script(script3)}\n")

    script4 = "k = 100  # Ends with assignment"
    print(f"Script 4:\n{script4}\nResult: {execute_script(script4)}\n")

    script5 = "print('Hello from script')\n7 * 7"
    print(f"Script 5 (output will appear above result line):\n{script5}")
    print(f"Result: {execute_script(script5)}\n")

    script_empty = ""
    print(f"Script Empty:\n'{script_empty}'\nResult: {execute_script(script_empty)}\n")

    script_only_comment = "# This is a comment\n# and another one"
    print(f"Script Only Comment:\n'{script_only_comment}'\nResult: {execute_script(script_only_comment)}\n")

    script_just_expression = "99 + 1"
    print(f"Script Just Expression:\n'{script_just_expression}'\nResult: {execute_script(script_just_expression)}\n")

    # Error handling examples
    script_syntax_error = "a = 1 +"
    print(f"Script Syntax Error:\n{script_syntax_error}")
    try:
        execute_script(script_syntax_error)
    except SyntaxError as e:
        print(f"Caught expected syntax error: {e}\n")

    script_runtime_error = "'text' + 123" # TypeError
    print(f"Script Runtime Error (TypeError):\n{script_runtime_error}")
    try:
        execute_script(script_runtime_error)
    except TypeError as e:
        print(f"Caught expected runtime error: {e}\n")

    script_zero_division = "b = 0\n10 / b"
    print(f"Script Runtime Error (ZeroDivisionError):\n{script_zero_division}")
    try:
        execute_script(script_zero_division)
    except ZeroDivisionError as e:
        print(f"Caught expected runtime error: {e}\n")

    # Security tests (these should fail with NameError or similar)
    script_file_access = "open('test.txt', 'w')"
    print(f"Script File Access (should fail):\n{script_file_access}")
    try:
        execute_script(script_file_access)
    except NameError as e:
        print(f"Caught expected security error (NameError for 'open'): {e}\n")
    except Exception as e:
        print(f"Caught UNEXPECTED error for file access: {e}\n")

    script_import_os = "import os\nos.getcwd()"
    print(f"Script Import OS (should fail):\n{script_import_os}")
    try:
        # 'import' itself relies on '__import__', which is not in SAFE_BUILTINS
        execute_script(script_import_os)
    except NameError as e: # '__import__' not found, or 'os' not found after failed import
        print(f"Caught expected security error (NameError, likely for '__import__'): {e}\n")
    except Exception as e:
        print(f"Caught UNEXPECTED error for import os: {e}\n")

    script_eval_inside = "eval('1+1')"
    print(f"Script eval inside (should fail):\n{script_eval_inside}")
    try:
        execute_script(script_eval_inside)
    except NameError as e: # 'eval' should not be defined
        print(f"Caught expected security error (NameError for 'eval'): {e}\n")
    except Exception as e:
        print(f"Caught UNEXPECTED error for eval inside: {e}\n")

    script_exec_inside = "exec('a=1')"
    print(f"Script exec inside (should fail):\n{script_exec_inside}")
    try:
        execute_script(script_exec_inside)
    except NameError as e: # 'exec' should not be defined
        print(f"Caught expected security error (NameError for 'exec'): {e}\n")
    except Exception as e:
        print(f"Caught UNEXPECTED error for exec inside: {e}\n")

    script_dangerous_getattr = "x = ().__class__.__bases__[0].__subclasses__()[0].__init__.__globals__['sys']"
    print(f"Script dangerous getattr-like (should fail if type/getattr restricted or sys not available):\n{script_dangerous_getattr}")
    try:
        # This kind of attack is harder to block without more sophisticated sandboxing
        # or removing type/getattr. Current setup might allow parts of it if type is available.
        # If 'type' is in SAFE_BUILTINS, this can be an issue.
        # If 'sys' is not in restricted_globals, the final access will fail.
        execute_script(script_dangerous_getattr)
    except Exception as e: # Expecting NameError for 'sys' or AttributeError if path is broken
        print(f"Caught error (expected due to restricted env): {e}\n")
