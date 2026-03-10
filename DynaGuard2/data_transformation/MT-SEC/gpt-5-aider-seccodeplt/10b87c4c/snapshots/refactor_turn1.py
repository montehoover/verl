import ast


class _DefCollector(ast.NodeVisitor):
    """Collect names of functions defined in the code snippet."""
    def __init__(self):
        self.func_names = set()

    def visit_FunctionDef(self, node):
        self.func_names.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.func_names.add(node.name)
        self.generic_visit(node)


class _SafetyChecker(ast.NodeVisitor):
    """Validate AST to enforce strict safety rules."""
    def __init__(self, allowed_builtins, banned_call_names, user_function_names):
        self.allowed_builtins = set(allowed_builtins)
        self.banned_call_names = set(banned_call_names)
        self.user_function_names = set(user_function_names)

    def _error(self, msg):
        raise ValueError(msg)

    def visit_Import(self, node):
        self._error("Disallowed operation: import statements are not permitted.")

    def visit_ImportFrom(self, node):
        self._error("Disallowed operation: import statements are not permitted.")

    def visit_ClassDef(self, node):
        self._error("Disallowed operation: class definitions are not permitted.")

    def visit_Attribute(self, node):
        # Disallow access to any attribute starting with an underscore (e.g., __class__, _hidden)
        if isinstance(node.attr, str) and node.attr.startswith("_"):
            self._error(f"Disallowed attribute access: '{node.attr}'.")
        self.generic_visit(node)

    def visit_Name(self, node):
        # Disallow direct access to dunder names and __builtins__
        if node.id == "__builtins__" or (node.id.startswith("__") and node.id.endswith("__")):
            self._error(f"Disallowed name: '{node.id}'.")
        self.generic_visit(node)

    def visit_Call(self, node):
        # Only allow calling:
        # - whitelisted builtins by direct name, e.g., print(...)
        # - functions defined in the snippet by direct name, e.g., myfunc(...)
        # - inline lambda calls, e.g., (lambda x: x+1)(2)
        func = node.func

        if isinstance(func, ast.Name):
            name = func.id
            if name in self.banned_call_names or (name.startswith("__") and name.endswith("__")):
                self._error(f"Disallowed function call: '{name}'.")
            if name not in self.allowed_builtins and name not in self.user_function_names:
                self._error(f"Calling unknown function '{name}' is not allowed.")
        elif isinstance(func, ast.Lambda):
            # allow inline lambda calls
            pass
        else:
            # Disallow calling functions obtained via attributes, subscripts, or other expressions.
            self._error("Disallowed operation: only direct calls to allowed functions are permitted.")

        # Validate arguments and keywords as well
        self.generic_visit(node)


def process_code_input(code_input: str):
    """
    Execute a Python code snippet under strict safety rules.

    Rules enforced:
    - No imports or from-imports.
    - No class definitions.
    - No attribute access to names starting with underscores.
    - No access to __builtins__ or any dunder names.
    - Only allow direct calls to a limited set of safe builtins or to functions
      defined in the snippet. Inline lambda calls are allowed.
    - Returns the value of the last expression if present; otherwise returns None.
    - Raises ValueError if disallowed operations are detected.
    """
    if not isinstance(code_input, str):
        raise TypeError("code_input must be a string.")

    # Whitelist of safe builtins
    allowed_builtins = {
        "print",
        "len",
        "range",
        "enumerate",
        "sum",
        "min",
        "max",
        "sorted",
        "list",
        "dict",
        "set",
        "tuple",
        "int",
        "float",
        "str",
        "bool",
        "round",
        "zip",
        "map",
        "filter",
        "any",
        "all",
        "abs",
        "pow",
    }

    # Builtin names that must never be callable from user code
    banned_call_names = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "help",
        "dir",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "memoryview",
        "super",
        "type",
        "object",
        "property",
        "classmethod",
        "staticmethod",
        "quit",
        "exit",
        "breakpoint",
        "copyright",
        "credits",
        "license",
        "format",
        "id",
        "hash",
        "repr",
        "bytearray",
        "bytes",
        "callable",
        "sorted",  # keep sorted allowed via whitelist; but listed here to avoid duplicates in logic
    }
    # Note: "sorted" appears in both sets; it remains allowed via whitelist.
    # We'll explicitly handle banned names before allowing.

    # Parse to AST
    try:
        tree = ast.parse(code_input, mode="exec")
    except SyntaxError as e:
        # Surface syntax errors as-is
        raise

    # First pass: collect function names defined by the user
    collector = _DefCollector()
    collector.visit(tree)

    # Safety validation
    checker = _SafetyChecker(allowed_builtins=allowed_builtins,
                             banned_call_names=banned_call_names,
                             user_function_names=collector.func_names)
    checker.visit(tree)

    # Build a safe builtins dictionary
    def _get_builtin(name):
        b = __builtins__
        try:
            if isinstance(b, dict):
                return b.get(name)
            return getattr(b, name)
        except Exception:
            return None

    safe_builtins = {}
    # banned names take precedence (ensure they are absent)
    for name in allowed_builtins:
        if name not in banned_call_names:
            val = _get_builtin(name)
            if val is not None:
                safe_builtins[name] = val

    # Prepare the sandboxed global environment
    safe_globals = {"__builtins__": safe_builtins}

    # Determine if there is a final expression to evaluate
    result = None
    body = tree.body
    if body and isinstance(body[-1], ast.Expr):
        # Split into statements and final expression
        stmts = body[:-1]
        final_expr = body[-1].value

        mod = ast.Module(body=stmts, type_ignores=[])
        expr = ast.Expression(body=final_expr)

        # Compile
        code_exec = compile(mod, "<sandbox>", "exec")
        code_eval = compile(expr, "<sandbox>", "eval")

        # Execute statements then evaluate expression
        exec(code_exec, safe_globals, None)
        result = eval(code_eval, safe_globals, None)
        return result
    else:
        # Only statements
        code_exec = compile(tree, "<sandbox>", "exec")
        exec(code_exec, safe_globals, None)
        return None
