import ast
import io
import contextlib
import builtins # To get actual builtin functions

# Whitelist of AST node types allowed in the user's script.
# Based on Python 3.8+ AST nodes. More restrictive is generally safer.
_ALLOWED_NODE_TYPES = {
    ast.Module, ast.Expr, ast.Constant, # For literals: numbers, strings, True, False, None
    ast.Name, ast.Load, ast.Store,      # For variables and contexts
    ast.Assign, ast.AugAssign,          # Assignment operations
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, # Operators
    ast.Call,                           # Function calls (restricted)
    ast.If, ast.For, ast.While,         # Control flow statements
    ast.Break, ast.Continue, ast.Pass,  # Control flow keywords
    ast.List, ast.Tuple, ast.Dict, ast.Set, # Data structure literals
    ast.Subscript, ast.Slice,           # Indexing and slicing
    ast.Starred,                        # For *args in calls or unpacking (e.g., print(*my_list))
    ast.keyword,                        # For keyword arguments in calls (e.g., print(..., sep='-'))
}

# Whitelist of built-in function names that can be called by the user script.
_ALLOWED_BUILTIN_FUNCTION_NAMES = {
    'print', 'len', 'abs', 'round', 'max', 'min', 'sum', 'range',
    'str', 'int', 'float', 'bool',
    'list', 'dict', 'tuple', 'set',
    # Other safe builtins like 'all', 'any', 'sorted', 'zip', 'enumerate' can be added if needed.
}

# Construct the __builtins__ dictionary for the exec environment.
# This dictionary will be the only part of __builtins__ accessible to the user script.
# It only includes callable functions from the whitelist.
_SAFE_BUILTINS_FOR_USER_SCRIPT = {
    name: getattr(builtins, name)
    for name in _ALLOWED_BUILTIN_FUNCTION_NAMES
    if hasattr(builtins, name) and callable(getattr(builtins, name))
}
# As a safeguard, ensure 'print' is available if it's a standard builtin,
# as it's fundamental for producing output.
if 'print' not in _SAFE_BUILTINS_FOR_USER_SCRIPT and hasattr(builtins, 'print'):
    _SAFE_BUILTINS_FOR_USER_SCRIPT['print'] = builtins.print


def restricted_run_code(run_code: str):
    """
    Executes a Python script specified by the user, limiting it to safe operations.

    Args:
        run_code: A string containing the user's Python script.

    Returns:
        The output of the script (stdout, if applicable), or None if no output.

    Raises:
        ValueError: If the script attempts prohibited actions, has syntax issues,
                    or encounters a runtime error during execution.
    """
    try:
        # Parse the user's code into an Abstract Syntax Tree (AST)
        # filename='<user_script>' helps in tracebacks if any occur from ast.parse itself.
        tree = ast.parse(run_code, filename='<user_script>')
    except SyntaxError as e:
        raise ValueError(f"Syntax error in script: {e}")

    # Validate the AST: walk through all nodes and check against whitelists
    for node in ast.walk(tree):
        node_type = type(node)

        if node_type not in _ALLOWED_NODE_TYPES:
            # Provide specific messages for common prohibited operations for better user feedback
            if node_type == ast.Import or node_type == ast.ImportFrom:
                raise ValueError("Imports are prohibited.")
            if node_type == ast.Attribute: # e.g., obj.attr or obj.method()
                raise ValueError("Attribute access (e.g., 'obj.attr' or 'obj.method()') is prohibited.")
            if node_type == ast.FunctionDef or node_type == ast.ClassDef:
                raise ValueError("Defining functions or classes is prohibited.")
            if node_type == ast.Lambda:
                raise ValueError("Lambda functions are prohibited.")
            if node_type == ast.Delete:
                raise ValueError("Deleting variables or attributes (del statement) is prohibited.")
            if node_type == ast.Try: # try-except, try-finally
                raise ValueError("Try-except blocks and try-finally are prohibited.")
            if node_type == ast.With: # `with` statement, often for file I/O or locks
                raise ValueError("The 'with' statement is prohibited.")
            if node_type == ast.Raise: # raise Exception()
                raise ValueError("Raising exceptions (raise statement) is prohibited.")
            if node_type == ast.Assert: # assert condition
                raise ValueError("Assert statements are prohibited.")
            if node_type == ast.Global or node_type == ast.Nonlocal:
                raise ValueError("'global' and 'nonlocal' statements are prohibited.")
            if node_type == ast.Yield or node_type == ast.YieldFrom:
                raise ValueError("Generator functions ('yield', 'yield from') are prohibited.")
            # Generic message for other disallowed AST node types
            raise ValueError(f"Prohibited language feature: {node_type.__name__}")

        # Special checks for Call nodes (function calls)
        if isinstance(node, ast.Call):
            # The function being called must be a simple name (e.g., print),
            # not a complex expression (e.g., (lambda x:x)())
            if not isinstance(node.func, ast.Name):
                # This also implicitly disallows calling methods on objects (e.g. "mystring".upper()),
                # as `node.func` would be an `ast.Attribute` in such cases,
                # which is already disallowed by the `ast.Attribute` check above.
                # This check is an additional safeguard for other non-Name call targets.
                raise ValueError("Indirect function calls or calling methods on objects is prohibited.")
            
            function_name = node.func.id
            if function_name not in _ALLOWED_BUILTIN_FUNCTION_NAMES:
                raise ValueError(f"Prohibited function call: {function_name}(...). "
                                 f"Only whitelisted built-in functions can be called.")
            # Arguments (node.args, node.keywords) are themselves AST nodes
            # and will be recursively validated by ast.walk.

    # If AST validation passes, compile the tree into a code object
    try:
        # Compiling from the tree ensures consistency with the validated AST.
        code_obj = compile(tree, filename='<user_script>', mode='exec')
    except Exception as e:
        # This catch is a safeguard for rare compilation errors on an already parsed/validated AST.
        raise ValueError(f"Error compiling script after validation: {e}")

    # Prepare the execution environment (globals and locals)
    # The script will only have access to the builtins defined in _SAFE_BUILTINS_FOR_USER_SCRIPT.
    # All other globals are absent, preventing access to modules like 'os' or 'sys' unless explicitly passed.
    user_globals = {"__builtins__": _SAFE_BUILTINS_FOR_USER_SCRIPT}
    user_locals = {}  # The script will use this dictionary for its local variables.

    # Capture stdout (e.g., from print() statements)
    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code_obj, user_globals, user_locals)
    except Exception as e:
        # Catch and wrap runtime errors from the user's script (e.g., ZeroDivisionError, NameError)
        # This ensures that errors from the script are also reported as ValueError,
        # fitting the function's error contract.
        raise ValueError(f"Error during script execution: {type(e).__name__}: {e}")

    output = stdout_capture.getvalue()
    
    # Return the captured stdout content. If nothing was printed, output is an empty string.
    # Per function spec: "The output of the script (if applicable), or None."
    if output:
        return output
    else:
        return None
