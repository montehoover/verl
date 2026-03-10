import ast

# Define a whitelist of allowed built-in functions
ALLOWED_BUILTINS = {
    'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
    'callable', 'chr', 'complex', 'dict', 'divmod', 'enumerate', 'filter',
    'float', 'format', 'frozenset', 'getattr', 'hasattr', 'hash', 'hex',
    'id', 'int', 'isinstance', 'issubclass', 'iter', 'len', 'list', 'map',
    'max', 'min', 'next', 'oct', 'ord', 'pow', 'print', 'range', 'repr',
    'reversed', 'round', 'set', 'slice', 'sorted', 'str', 'sum', 'super',
    'tuple', 'type', 'vars', 'zip',
    # Math functions that are generally safe
    'round', 
}

# Define a blacklist of AST node types that are considered unsafe
FORBIDDEN_NODES = (
    ast.Import, ast.ImportFrom, # Disallow imports
    # Potentially dangerous builtins if not handled carefully or if __builtins__ is exposed directly
    # ast.Call with func.id being 'eval', 'exec', 'open', 'compile', '__import__'
    # ast.Attribute with attr being 'system' from 'os' module (if os was somehow imported)
    # For simplicity, we'll block all direct calls to known dangerous functions by name.
)

# Names of functions/attributes that should not be callable
FORBIDDEN_CALLS = {
    'eval', 'exec', 'open', 'compile', '__import__', 'exit', 'quit',
    'input', # Can hang the process
    # File system access
    'system', # os.system
    # Attributes that could lead to arbitrary code execution or sensitive info
    '__class__', '__subclasses__', '__bases__', '__globals__', '__code__',
    '__closure__', '__func__', '__self__', '__dict__', '__mro__',
    '__builtins__', # Direct access to all builtins
    '__file__', '__path__', '__package__', '__loader__', '__spec__',
    'gi_frame', 'gi_code', # Generator internals
    'f_locals', 'f_globals', 'f_builtins', 'f_code', # Frame internals
}


class SafeASTChecker(ast.NodeVisitor):
    def visit(self, node):
        node_type = type(node)
        if node_type in FORBIDDEN_NODES:
            raise ValueError(f"Forbidden operation: {node_type.__name__} is not allowed.")
        
        if isinstance(node, ast.Call):
            # Check for calls to forbidden functions by name
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                raise ValueError(f"Forbidden function call: {node.func.id}() is not allowed.")
            # Check for calls to forbidden attributes (e.g., some_object.system())
            if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_CALLS:
                 raise ValueError(f"Forbidden method/attribute call: .{node.func.attr}() is not allowed.")

        if isinstance(node, ast.Attribute):
            # Prevent access to forbidden attributes
            if node.attr in FORBIDDEN_CALLS:
                raise ValueError(f"Forbidden attribute access: .{node.attr} is not allowed.")
        
        # Check for __import__ in Name nodes (though ast.Import should catch it)
        if isinstance(node, ast.Name) and node.id == '__import__':
            raise ValueError("Forbidden operation: __import__ is not allowed.")

        super().visit(node)


def execute_python_code(snippet_code: str):
    """
    Safely executes a string of Python code.

    Args:
        snippet_code: A string containing Python code.

    Returns:
        The resulting value if the code's final expression produces output,
        or None if the code does not end with an expression or the expression
        evaluates to None.

    Raises:
        ValueError: If the code attempts forbidden operations, is empty,
                    contains only whitespace, or has other parsing/validation errors.
        SyntaxError: If the snippet_code has Python syntax errors.
        NameError: If the snippet_code contains undefined variables.
        TypeError: If operations are attempted on incompatible types.
        Exception: For other evaluation/execution errors not caught as ValueError.
    """
    stripped_code = snippet_code.strip()
    if not stripped_code:
        raise ValueError("Input code snippet is empty or contains only whitespace.")

    try:
        code_ast = ast.parse(stripped_code)
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax: {e}") from e # Re-raise as ValueError or keep SyntaxError

    if not code_ast.body:
        # This can happen if the code is only comments after stripping
        raise ValueError("Input code snippet does not contain executable code.")

    # Validate the AST for forbidden operations
    checker = SafeASTChecker()
    try:
        checker.visit(code_ast)
    except ValueError: # Re-raise if checker raises ValueError
        raise

    # Prepare a safe global and local scope for execution
    safe_globals = {"__builtins__": {
        name: __builtins__[name] for name in ALLOWED_BUILTINS if name in __builtins__
    }}
    local_vars = {}

    try:
        # Execute all statements except possibly the last one (if it's an expression)
        if len(code_ast.body) > 1:
            exec_ast_module = ast.Module(code_ast.body[:-1], type_ignores=[])
            exec_code_obj = compile(exec_ast_module, filename="<snippet_exec>", mode="exec")
            exec(exec_code_obj, safe_globals, local_vars)
        
        last_node = code_ast.body[-1]

        # If the last node is an expression, evaluate it and return its result
        if isinstance(last_node, ast.Expr):
            eval_ast_expression = ast.Expression(last_node.value)
            eval_code_obj = compile(eval_ast_expression, filename="<snippet_eval>", mode="eval")
            result = eval(eval_code_obj, safe_globals, local_vars)
            return result
        else:
            # If the last node is a statement (e.g., assignment, def, pass),
            # execute it but return None as there's no expression value.
            # We need to compile and exec this single last statement.
            final_exec_ast_module = ast.Module([last_node], type_ignores=[])
            final_exec_code_obj = compile(final_exec_ast_module, filename="<snippet_final_exec>", mode="exec")
            exec(final_exec_code_obj, safe_globals, local_vars)
            return None # No expression value to return

    except (NameError, TypeError) as e: # Let these specific errors propagate
        raise
    except ValueError: # Re-raise ValueErrors (e.g. from checker, or our own)
        raise
    except Exception as e:
        # Catch other potential runtime errors during exec/eval
        raise ValueError(f"Error during code execution: {e}") from e


if __name__ == '__main__':
    print("--- Basic Safe Operations ---")
    print(f"Result of '2 + 3': {execute_python_code('2 + 3')}")
    print(f"Result of 'abs(-5)': {execute_python_code('abs(-5)')}")
    op_str1 = 'a = 5; b = 10; a * b + 2'
    print(f"Result of '{op_str1}': {execute_python_code(op_str1)}")
    
    op_str2 = """
    x = [1, 2, 3]
    len(x)
    """
    print(f"Result of multi-line:\n{op_str2.strip()}\nOutput: {execute_python_code(op_str2)}")

    op_str3 = "my_var = 'hello world'; my_var"
    print(f"Result of '{op_str3}': {execute_python_code(op_str3)}")

    op_str4 = "c = 10" # Assignment as last statement
    print(f"Result of '{op_str4}': {execute_python_code(op_str4)}") # Should be None

    op_str5 = "pass" # Pass statement
    print(f"Result of '{op_str5}': {execute_python_code(op_str5)}") # Should be None

    print("\n--- Error Handling & Forbidden Operations ---")
    test_cases_fail = [
        ("1 / 0", ZeroDivisionError), # Specific error, not ValueError by default from execute_python_code
        ("import os", ValueError),
        ("from math import sqrt", ValueError),
        ("eval('1+1')", ValueError),
        ("exec('a=1')", ValueError),
        ("open('file.txt', 'w')", ValueError),
        ("__import__('os').system('echo unsafe')", ValueError),
        ("a = {}; a.__class__", ValueError), # Accessing forbidden attribute
        ("print.__globals__", ValueError), # Accessing forbidden attribute on allowed builtin
        ("().__class__.__bases__[0].__subclasses__()[0].__init__.__globals__['sys'].exit(1)", ValueError), # complex attack
        ("a + 3", NameError), # Undefined variable
        ("2 +", ValueError), # Syntax error (re-raised as ValueError by the function)
        ("", ValueError), # Empty string
        ("   ", ValueError), # Whitespace only
        ("# just a comment", ValueError), # Comment only
        ("del x", ValueError), # ast.Delete is not explicitly forbidden but good to test if it passes checker
                               # Current checker doesn't explicitly block ast.Delete, but it's not an Expr.
                               # If it's the last statement, it will be exec'd and return None.
                               # If 'x' doesn't exist, it will raise NameError.
    ]

    for code, error_type in test_cases_fail:
        try:
            print(f"\nTesting: {code}")
            execute_python_code(code)
            print(f"  !!! UNEXPECTED SUCCESS for: {code}")
        except error_type as e:
            print(f"  OK: Got expected {error_type.__name__}: {e}")
        except Exception as e:
            print(f"  !!! WRONG EXCEPTION for: {code}. Got {type(e).__name__}: {e}")

    print("\n--- Test case for del (should be None or NameError) ---")
    try:
        print(f"Result of 'x=5; del x': {execute_python_code('x=5; del x')}") # Should be None
        execute_python_code('del y') # Should raise NameError
    except NameError as e:
        print(f"  OK: Got expected NameError for 'del y': {e}")
    except Exception as e:
        print(f"  !!! UNEXPECTED error for del tests: {e}")

    print("\n--- Test case for print() (should return None) ---")
    print(f"Result of 'print(\"hello\")': {execute_python_code('''print("hello")''')}")
