import ast

# A whitelist of built-in functions that are considered safe to use.
_ALLOWED_BUILTINS = {
    'abs': abs, 'all': all, 'any': any, 'ascii': ascii, 'bin': bin, 'bool': bool,
    'bytearray': bytearray, 'bytes': bytes, 'callable': callable, 'chr': chr,
    'complex': complex, 'dict': dict, 'divmod': divmod, 'enumerate': enumerate,
    'filter': filter, 'float': float, 'format': format, 'frozenset': frozenset,
    'hash': hash, 'hex': hex, 'id': id, 'int': int, 'isinstance': isinstance,
    'issubclass': issubclass, 'iter': iter, 'len': len, 'list': list, 'map': map,
    'max': max, 'memoryview': memoryview, 'min': min, 'next': next, 'object': object,
    'oct': oct, 'ord': ord, 'pow': pow, 'print': print, 'property': property,
    'range': range, 'repr': repr, 'reversed': reversed, 'round': round, 'set': set,
    'slice': slice, 'sorted': sorted, 'str': str, 'sum': sum, 'super': super,
    'tuple': tuple, 'type': type, 'zip': zip,
    # Potentially harmful built-ins like open, eval, exec, input, getattr, setattr, delattr, compile, __import__
    # are intentionally omitted.
}

class _SecurityValidator(ast.NodeVisitor):
    """
    Traverses an AST to ensure it doesn't contain forbidden operations.
    Forbidden operations include:
    - Imports
    - Calls to dangerous functions (eval, exec, open, etc.)
    - Access to private/special attributes (those starting with '_')
    - Use of 'global' or 'nonlocal' keywords
    - Direct access to '__builtins__'
    """
    FORBIDDEN_CALLS = {'eval', 'exec', 'open', 'input', '__import__', 'compile', 'getattr', 'setattr', 'delattr'}

    def __init__(self):
        super().__init__()
        self._errors = []

    def _report_error(self, node, message):
        self._errors.append(f"Error at line {getattr(node, 'lineno', 'unknown')}, col {getattr(node, 'col_offset', 'unknown')}: {message}")

    def visit_Import(self, node):
        self._report_error(node, "Import statements (import) are forbidden.")
        # Do not call generic_visit to stop further processing of this branch if needed,
        # though for imports, the node itself is the issue.

    def visit_ImportFrom(self, node):
        self._report_error(node, "Import statements (from ... import) are forbidden.")

    def visit_Call(self, node):
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
            # Check for __builtins__.dangerous_func()
            if isinstance(node.func.value, ast.Name) and node.func.value.id == '__builtins__':
                if func_name in self.FORBIDDEN_CALLS or func_name.startswith('_'):
                    self._report_error(node, f"Calling forbidden method '{func_name}' on '__builtins__'.")

        if func_name in self.FORBIDDEN_CALLS:
            self._report_error(node, f"Call to forbidden function '{func_name}'.")
        
        self.generic_visit(node) # Continue to check arguments and the function expression itself

    def visit_Attribute(self, node):
        # Disallow access to attributes starting with an underscore (e.g., obj._secret, obj.__dict__)
        if node.attr.startswith('_'):
            self._report_error(node, f"Access to private or special attribute '{node.attr}' is forbidden.")
        self.generic_visit(node) # Check the base of the attribute access (node.value)

    def visit_Name(self, node):
        # Disallow direct use of __builtins__ as a name.
        # This prevents __builtins__.open(...) if __builtins__ was somehow exposed.
        if node.id == '__builtins__':
            self._report_error(node, "Direct access to '__builtins__' is forbidden.")
        self.generic_visit(node)

    def visit_Global(self, node):
        self._report_error(node, "'global' keyword is forbidden.")
        self.generic_visit(node)

    def visit_Nonlocal(self, node):
        self._report_error(node, "'nonlocal' keyword is forbidden.")
        self.generic_visit(node)

    def validate(self, tree):
        self._errors = []
        self.visit(tree)
        if self._errors:
            raise ValueError("Forbidden operations found in code:\n" + "\n".join(self._errors))


def _parse_and_validate_code(snippet_code: str) -> ast.AST:
    """
    Parses the Python code string into an AST and validates it for forbidden operations.

    Args:
        snippet_code: The user-provided Python code string.

    Returns:
        The validated Abstract Syntax Tree (AST).

    Raises:
        ValueError: If the snippet contains invalid syntax or forbidden operations.
    """
    # 1. Parse the code into an AST
    try:
        tree = ast.parse(snippet_code, filename='<user_snippet>')
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax: {e}") from e

    # 2. Validate the AST for forbidden operations
    validator = _SecurityValidator()
    try:
        validator.validate(tree)
    except ValueError:  # Re-raise, it already has detailed messages
        raise
    return tree


def _execute_ast(tree: ast.AST, safe_globals: dict):
    """
    Executes a validated AST in a restricted environment.

    Args:
        tree: The validated Abstract Syntax Tree (AST) to execute.
        safe_globals: A dictionary of globals to use for execution,
                      typically containing restricted builtins.

    Returns:
        The resulting value if the code is a single expression, or None otherwise.

    Raises:
        ValueError: If an error occurs during compilation or execution.
    """
    result = None

    if not tree.body:  # Empty snippet
        return None

    # Determine execution mode (eval for single expression, exec for statements)
    # and compile the code.
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
        # The snippet is a single expression. Compile for 'eval'.
        # ast.Expression wraps the actual expression node (tree.body[0].value).
        eval_node = ast.Expression(body=tree.body[0].value)
        try:
            code_obj = compile(eval_node, filename='<user_snippet>', mode='eval')
        except Exception as e:
            raise ValueError(f"Error compiling expression: {e}") from e
        
        # Execute the compiled code
        try:
            result = eval(code_obj, safe_globals, {})  # Use an empty dict for locals
        except Exception as e:
            raise ValueError(f"Error during execution of expression: {e}") from e
    else:
        # The snippet is a block of statements or more complex. Compile for 'exec'.
        try:
            code_obj = compile(tree, filename='<user_snippet>', mode='exec')
        except Exception as e:
            raise ValueError(f"Error compiling statements: {e}") from e

        # Execute the compiled code
        try:
            # For 'exec', results are typically side effects (e.g., print) or
            # modifications to the locals_dict. We return None as per requirements.
            exec_locals = {} 
            exec(code_obj, safe_globals, exec_locals)
            result = None 
        except Exception as e:
            raise ValueError(f"Error during execution of statements: {e}") from e
            
    return result


def execute_python_code(snippet_code: str):
    """
    Executes a string of Python code in a restricted environment.

    Args:
        snippet_code: The user-provided Python code to be executed.

    Returns:
        The resulting value if the code is a single expression, or None otherwise.

    Raises:
        ValueError: If the snippet attempts forbidden operations, contains invalid syntax,
                    or an error occurs during execution.
    """
    # 1. Parse and validate the code
    # This will raise ValueError on issues, propagating it up.
    tree = _parse_and_validate_code(snippet_code)

    # 2. Prepare the execution environment
    # Provide a restricted set of builtins. Make a copy to prevent modification.
    safe_globals = {"__builtins__": _ALLOWED_BUILTINS.copy()}
    
    # 3. Execute the AST
    # This will raise ValueError on issues, propagating it up.
    result = _execute_ast(tree, safe_globals)
            
    return result
