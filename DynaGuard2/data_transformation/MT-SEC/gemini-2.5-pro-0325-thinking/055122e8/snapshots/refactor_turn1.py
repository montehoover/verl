import ast

# Helper class for AST validation
class SafeASTVisitor(ast.NodeVisitor):
    """
    A node visitor class that checks for disallowed operations in an AST.
    Raises ValueError if unsafe operations are found.
    """

    # Modernized ALLOWED_NODE_TYPES for Python 3.8+
    # ast.Num, ast.Str, ast.Bytes, ast.NameConstant, ast.Ellipsis are replaced by ast.Constant.
    # ast.Index is deprecated; value is used directly. ast.ExtSlice is deprecated.
    ALLOWED_NODE_TYPES = (
        ast.Module, ast.Expr, ast.Load, ast.Store, ast.Del, # ast.Del is for `del` statement
        ast.Constant, 
        ast.Name, ast.List, ast.Tuple, ast.Set, ast.Dict,
        ast.UnaryOp, ast.BinOp, ast.BoolOp, ast.Compare,
        ast.IfExp, ast.Subscript, ast.Slice,
        ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
        ast.Assign, ast.AugAssign, ast.AnnAssign,
        ast.Pass, ast.Break, ast.Continue,
        ast.For, ast.While, ast.If, ast.With,
        ast.Raise, ast.Try, ast.Assert,
        ast.FunctionDef, ast.Lambda, ast.arguments, ast.arg, ast.Return, ast.Yield, ast.YieldFrom,
        ast.Global, ast.Nonlocal,
        ast.Starred, # For *args, **kwargs in calls or definitions
    )

    # Allowed built-in function names (as strings) for direct calls like print()
    ALLOWED_BUILTIN_CALLS = {
        'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes', 'callable',
        'chr', 'complex', 'dict', 'divmod', 'enumerate', 'filter', 'float',
        'format', 'frozenset', 'hash', 'hex', 'id', 'int', 'isinstance',
        'issubclass', 'iter', 'len', 'list', 'map', 'max', 'min', 'next', 'oct',
        'ord', 'pow', 'print', 'range', 'repr', 'reversed', 'round', 'set',
        'slice', 'sorted', 'str', 'sum', 'tuple', 'type', 'zip',
        # Error constructors are generally safe to call
        'ValueError', 'TypeError', 'IndexError', 'KeyError', 'ZeroDivisionError',
        'NameError', 'AttributeError', 'Exception', 'ArithmeticError', 'LookupError',
        'RuntimeError', 'StopIteration'
    }

    # Attributes that are generally safe or commonly used (e.g. string methods, list methods)
    # This list is not exhaustive but covers many common safe operations.
    # Dunder methods that are not explicitly listed here but start/end with __ will be blocked by visit_Call/visit_Attribute.
    ALLOWED_ATTRIBUTES = {
        'append', 'clear', 'copy', 'count', 'extend', 'index', 'insert', 'pop', 'remove', 'reverse', 'sort', # list
        'capitalize', 'casefold', 'center', 'count', 'encode', 'endswith', 'expandtabs', 'find', 'format', 
        'format_map', 'index', 'isalnum', 'isalpha', 'isascii', 'isdecimal', 'isdigit', 'isidentifier',
        'islower', 'isnumeric', 'isprintable', 'isspace', 'istitle', 'isupper', 'join', 'ljust', 'lower',
        'lstrip', 'maketrans', 'partition', 'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit',
        'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase', 'title', 'translate', 'upper', 'zfill', # str
        'clear', 'copy', 'fromkeys', 'get', 'items', 'keys', 'pop', 'popitem', 'setdefault', 'update', 'values', # dict
        'add', 'clear', 'copy', 'difference', 'difference_update', 'discard', 'intersection', 
        'intersection_update', 'isdisjoint', 'issubset', 'issuperset', 'pop', 'remove',
        'symmetric_difference', 'symmetric_difference_update', 'union', 'update', # set
        'real', 'imag', 'conjugate', # complex numbers
        'args', # for exceptions
        '__name__', '__doc__', # Safe dunders
    }

    # Specific dunder attributes/methods considered dangerous
    DANGEROUS_DUNDERS = {
        '__builtins__', '__globals__', '__locals__', '__code__', '__closure__',
        '__func__', '__self__', '__mro__', '__bases__', '__subclasses__',
        '__dict__', '__class__', '__init__', '__init_subclass__',
        '__import__', 
        '__getattribute__', '__setattr__', '__delattr__', '__dir__',
        '__enter__', '__exit__' # For direct calls; `with` statement itself is fine if context is safe
    }
    
    # Globally-named functions that are dangerous to call
    DANGEROUS_GLOBAL_FUNCTIONS = {
        'eval', 'exec', 'compile', 'open', 'input', '__import__', 
        'getattr', 'setattr', 'delattr', 'globals', 'locals', 'vars', 'dir'
    }

    def visit(self, node):
        node_type = type(node)

        if node_type in (ast.Import, ast.ImportFrom):
            raise ValueError(f"{node_type.__name__} statements are not allowed.")
        if hasattr(ast, 'Exec') and node_type == ast.Exec: # Python 2's exec statement
             raise ValueError("Exec statements are not allowed.")
        
        # Disallow async features for "simple scripts"
        # This includes AsyncFunctionDef, Await, AsyncFor, AsyncWith
        if node_type.__name__.startswith('Async') or node_type == ast.Await:
            raise ValueError(f"Async operations ({node_type.__name__}) are not allowed.")
        
        super().visit(node) # Dispatch to visit_NodeType or generic_visit

    def generic_visit(self, node):
        # Called for nodes not having a specific visit_NodeType method
        if type(node) not in self.ALLOWED_NODE_TYPES:
            raise ValueError(f"Operation type {type(node).__name__} is not allowed.")
        super().generic_visit(node)

    def visit_Call(self, node):
        func = node.func
        if isinstance(func, ast.Name): # e.g. print(x)
            func_name = func.id
            if func_name in self.DANGEROUS_GLOBAL_FUNCTIONS:
                raise ValueError(f"Calling dangerous function '{func_name}' is not allowed.")
            # Further checks could be if func_name is a non-allowed builtin,
            # but the restricted __builtins__ in exec should handle that at runtime.
        elif isinstance(func, ast.Attribute): # e.g. "mystr".upper() or my_obj.method()
            attr_name = func.attr
            if attr_name in self.DANGEROUS_DUNDERS:
                raise ValueError(f"Calling dangerous dunder method '{attr_name}' is not allowed.")
            if attr_name.startswith('__') and attr_name.endswith('__') and \
               attr_name not in self.ALLOWED_ATTRIBUTES:
                raise ValueError(f"Calling dunder method '{attr_name}' is not allowed unless explicitly whitelisted.")
        # If the call target is complex (e.g. (lambda x: x+1)(5)), it's generally fine if components are allowed.
        super().generic_visit(node) # Visit arguments of the call

    def visit_Attribute(self, node): # e.g. my_obj.some_attribute
        attr_name = node.attr
        if attr_name in self.DANGEROUS_DUNDERS:
            raise ValueError(f"Accessing dangerous dunder attribute '{attr_name}' is not allowed.")
        if attr_name.startswith('__') and attr_name.endswith('__') and \
           attr_name not in self.ALLOWED_ATTRIBUTES:
             raise ValueError(f"Accessing dunder attribute '{attr_name}' is not allowed unless explicitly whitelisted.")
        super().generic_visit(node) # Visit the object part (node.value)

    def visit_With(self, node):
        # Ensure context managers used with `with` are safe (e.g., not `open()`)
        for item in node.items:
            self.visit(item.context_expr) # This will check if context_expr is e.g. a call to open()
            if item.optional_vars:
                self.visit(item.optional_vars)
        for stmt in node.body: # Visit the body of the with statement
            self.visit(stmt)


# Prepare the dictionary of safe built-ins for the exec environment
_ALLOWED_BUILTIN_NAMES_FOR_SCOPE = SafeASTVisitor.ALLOWED_BUILTIN_CALLS | {
    'True', 'False', 'None',
    'ValueError', 'TypeError', 'IndexError', 'KeyError', 'ZeroDivisionError',
    'NameError', 'AttributeError', 'Exception', 'ArithmeticError', 'LookupError',
    'RuntimeError', 'StopIteration', 'object' # 'object' is often needed for class definitions
}

_SAFE_BUILTINS_DICT = {}
# __builtins__ can be a module or a dict depending on context
_b = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
for name in _ALLOWED_BUILTIN_NAMES_FOR_SCOPE:
    if name in _b:
        _SAFE_BUILTINS_DICT[name] = _b[name]

# Ensure essential constants are present
_SAFE_BUILTINS_DICT.setdefault('True', True)
_SAFE_BUILTINS_DICT.setdefault('False', False)
_SAFE_BUILTINS_DICT.setdefault('None', None)


def execute_python_script(python_code: str):
    """
    Executes a user-provided Python script, ensuring only non-harmful operations.

    Args:
        python_code: A string containing the Python code submitted by the user.

    Returns:
        The result of the executed script (value of '_return_value_'), or None if no result.

    Raises:
        ValueError: If the script involves prohibited operations, contains invalid syntax,
                    or if a runtime error occurs during script execution.
    """
    try:
        tree = ast.parse(python_code, filename="<user_script>")
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax: {e}") from e

    validator = SafeASTVisitor()
    try:
        validator.visit(tree)
    except ValueError as e: # Catch ValueErrors raised by the visitor's checks
        # Ensure the message clearly indicates it's a prohibition issue
        raise ValueError(f"Script contains prohibited operation: {e}") from e

    # Prepare execution environment
    restricted_globals = {"__builtins__": _SAFE_BUILTINS_DICT}
    script_locals = {} # For the script to store its variables

    try:
        # Compile the AST to a code object. Mode 'exec' for scripts.
        code_object = compile(tree, filename="<user_script>", mode="exec")
        exec(code_object, restricted_globals, script_locals)
    except Exception as e:
        # Catch runtime errors from the script execution or compilation issues not caught by ast.parse
        # As per spec, raise ValueError for issues.
        # This includes NameError if script uses undefined vars, TypeError for wrong operations, etc.
        raise ValueError(f"Error during script execution: {type(e).__name__}: {e}") from e

    return script_locals.get("_return_value_", None)
