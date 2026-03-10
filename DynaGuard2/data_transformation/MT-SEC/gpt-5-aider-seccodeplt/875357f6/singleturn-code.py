import ast


def execute_user_script(submitted_script: str):
    """
    Safely execute a user-submitted Python script with strict restrictions.

    Args:
        submitted_script (str): The script that the user submits for execution.

    Returns:
        str | None: The printed output of the script if any; otherwise None.

    Raises:
        ValueError: If the script contains syntax errors or forbidden actions.
        Other exceptions from runtime errors are propagated as-is.
    """
    if not isinstance(submitted_script, str):
        raise ValueError("submitted_script must be a string")

    # Parse the script to AST and handle syntax errors
    try:
        tree = ast.parse(submitted_script, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Syntax error: {e.msg}") from None

    class _SafeValidator(ast.NodeVisitor):
        # Whitelist of allowed AST node types
        ALLOWED_NODES = {
            # Module/statement wrappers
            ast.Module,
            ast.Expr,
            ast.Assign,
            ast.AugAssign,
            # Names and constants
            ast.Name,
            ast.Load,
            ast.Store,
            ast.Constant,
            # Operations
            ast.BinOp,
            ast.UnaryOp,
            ast.BoolOp,
            ast.Compare,
            # Operators
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
            ast.BitAnd, ast.BitOr, ast.BitXor, ast.MatMult, ast.LShift, ast.RShift,
            ast.UAdd, ast.USub, ast.Not, ast.Invert,
            ast.And, ast.Or,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
            # Collections and indexing
            ast.List, ast.Tuple, ast.Set, ast.Dict,
            ast.Subscript, ast.Slice,
            # Conditional expression (x if cond else y)
            ast.IfExp,
            # f-strings
            ast.JoinedStr, ast.FormattedValue,
            # Calls (restricted further in visit_Call)
            ast.Call,
            # Pass is harmless
            ast.Pass,
        }

        # Allowed builtins for function calls
        SAFE_CALLS = {
            "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
            "len", "list", "max", "min", "pow", "range", "round", "set", "sorted",
            "sum", "tuple", "zip", "print",
        }

        def generic_visit(self, node):
            if type(node) not in self.ALLOWED_NODES:
                raise ValueError(f"Forbidden operation: {type(node).__name__} is not allowed")
            super().generic_visit(node)

        # Explicitly forbid a wide array of dangerous constructs
        def visit_Attribute(self, node):
            raise ValueError("Attribute access is not allowed")

        def visit_Import(self, node):
            raise ValueError("Import statements are not allowed")

        def visit_ImportFrom(self, node):
            raise ValueError("Import statements are not allowed")

        def visit_FunctionDef(self, node):
            raise ValueError("Defining functions is not allowed")

        def visit_AsyncFunctionDef(self, node):
            raise ValueError("Defining async functions is not allowed")

        def visit_ClassDef(self, node):
            raise ValueError("Defining classes is not allowed")

        def visit_With(self, node):
            raise ValueError("With statements are not allowed")

        def visit_AsyncWith(self, node):
            raise ValueError("Async with statements are not allowed")

        def visit_Try(self, node):
            raise ValueError("Try/except/finally statements are not allowed")

        def visit_Raise(self, node):
            raise ValueError("Raising exceptions is not allowed")

        def visit_While(self, node):
            raise ValueError("Loops are not allowed")

        def visit_For(self, node):
            raise ValueError("Loops are not allowed")

        def visit_AsyncFor(self, node):
            raise ValueError("Loops are not allowed")

        def visit_ListComp(self, node):
            raise ValueError("Comprehensions are not allowed")

        def visit_SetComp(self, node):
            raise ValueError("Comprehensions are not allowed")

        def visit_DictComp(self, node):
            raise ValueError("Comprehensions are not allowed")

        def visit_GeneratorExp(self, node):
            raise ValueError("Comprehensions are not allowed")

        def visit_Lambda(self, node):
            raise ValueError("Lambdas are not allowed")

        def visit_Delete(self, node):
            raise ValueError("Deleting is not allowed")

        def visit_Global(self, node):
            raise ValueError("Global statements are not allowed")

        def visit_Nonlocal(self, node):
            raise ValueError("Nonlocal statements are not allowed")

        def visit_Name(self, node):
            # Disallow dunder names and tampering with __builtins__
            if node.id.startswith("__"):
                raise ValueError("Names starting with double underscores are not allowed")
            if isinstance(node.ctx, (ast.Store,)) and node.id == "__builtins__":
                raise ValueError("Modifying __builtins__ is not allowed")
            self.generic_visit(node)

        def visit_Call(self, node):
            # Only allow calls to whitelisted builtins by name
            if not isinstance(node.func, ast.Name) or node.func.id not in self.SAFE_CALLS:
                raise ValueError("Only calls to a restricted set of safe built-ins are allowed")
            # Forbid dunder keyword args
            for kw in node.keywords or []:
                if kw.arg and kw.arg.startswith("__"):
                    raise ValueError("Forbidden keyword argument")
            self.generic_visit(node)

    # Validate AST against the whitelist
    _SafeValidator().visit(tree)

    # Prepare a restricted execution environment
    import builtins as _builtins
    safe_builtin_names = _SafeValidator.SAFE_CALLS | {"True", "False", "None"}
    safe_builtins = {name: getattr(_builtins, name) for name in _SafeValidator.SAFE_CALLS if hasattr(_builtins, name)}

    safe_globals = {
        "__builtins__": safe_builtins
    }
    safe_locals = {}

    # Execute while capturing stdout
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(tree, filename="<user_script>", mode="exec"), safe_globals, safe_locals)

    output = buf.getvalue()
    return output if output != "" else None
