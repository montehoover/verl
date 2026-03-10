import ast


__all__ = ["execute_python_script"]


class _SafeScriptValidator(ast.NodeVisitor):
    """
    Validate that the AST contains only a safe subset of Python suitable for
    simple arithmetic and data manipulation without attribute access, calls,
    imports, or other potentially dangerous constructs.
    """

    _ALLOWED_BINOPS = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
    )

    _ALLOWED_UNARYOPS = (
        ast.UAdd,
        ast.USub,
        ast.Not,
    )

    _ALLOWED_BOOLOPS = (
        ast.And,
        ast.Or,
    )

    _ALLOWED_CMPOPS = (
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
    )

    def _ensure_allowed_target(self, target: ast.AST) -> None:
        # Only allow assignment to simple names or tuple/list of names
        if isinstance(target, ast.Name):
            # Disallow 'globals', 'locals', '__name__', etc. as assignment targets
            if target.id in {"__builtins__", "__import__", "__loader__", "__spec__", "__name__", "__package__"}:
                raise ValueError(f"Assignment to reserved name '{target.id}' is not allowed.")
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._ensure_allowed_target(elt)
            return
        # Disallow attribute or subscript assignment and any other complex targets
        raise ValueError("Only assignment to variable names (optionally unpacked) is allowed.")

    def visit_Module(self, node: ast.Module) -> None:
        for stmt in node.body:
            self.visit(stmt)

    def visit_Expr(self, node: ast.Expr) -> None:
        self.visit(node.value)

    def visit_Assign(self, node: ast.Assign) -> None:
        if not node.targets:
            raise ValueError("Invalid assignment.")
        for t in node.targets:
            self._ensure_allowed_target(t)
        self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._ensure_allowed_target(node.target)
        if not isinstance(node.op, self._ALLOWED_BINOPS):
            raise ValueError(f"Operator '{type(node.op).__name__}' is not allowed in augmented assignment.")
        self.visit(node.value)

    def visit_Name(self, node: ast.Name) -> None:
        # Allow variable usage and assignment; actual existence is a runtime concern.
        if isinstance(node.ctx, ast.Del):
            raise ValueError("Deletion is not allowed.")

    def visit_Constant(self, node: ast.Constant) -> None:
        # Allow numbers, strings, bytes, booleans, and None
        if isinstance(node.value, (int, float, str, bytes, bool, type(None))):
            return
        raise ValueError("Only simple literal constants are allowed.")

    # Support for older Python versions where numbers/strings may be represented differently
    if hasattr(ast, "Num"):
        def visit_Num(self, node: ast.AST) -> None:
            return  # numeric literals allowed

    if hasattr(ast, "Str"):
        def visit_Str(self, node: ast.AST) -> None:
            return  # string literals allowed

    def visit_List(self, node: ast.List) -> None:
        for elt in node.elts:
            self.visit(elt)

    def visit_Tuple(self, node: ast.Tuple) -> None:
        for elt in node.elts:
            self.visit(elt)

    def visit_Set(self, node: ast.Set) -> None:
        for elt in node.elts:
            self.visit(elt)

    def visit_Dict(self, node: ast.Dict) -> None:
        for k, v in zip(node.keys, node.values):
            if k is not None:
                self.visit(k)
            self.visit(v)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, self._ALLOWED_BINOPS):
            raise ValueError(f"Operator '{type(node.op).__name__}' is not allowed.")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, self._ALLOWED_UNARYOPS):
            raise ValueError(f"Unary operator '{type(node.op).__name__}' is not allowed.")
        self.visit(node.operand)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if not isinstance(node.op, self._ALLOWED_BOOLOPS):
            raise ValueError(f"Boolean operator '{type(node.op).__name__}' is not allowed.")
        for v in node.values:
            self.visit(v)

    def visit_Compare(self, node: ast.Compare) -> None:
        for op in node.ops:
            if not isinstance(op, self._ALLOWED_CMPOPS):
                raise ValueError(f"Comparison operator '{type(op).__name__}' is not allowed.")
        self.visit(node.left)
        for comp in node.comparators:
            self.visit(comp)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self.visit(node.value)
        self.visit(node.slice)

    def visit_Slice(self, node: ast.Slice) -> None:
        if node.lower:
            self.visit(node.lower)
        if node.upper:
            self.visit(node.upper)
        if node.step:
            self.visit(node.step)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        for value in node.values:
            self.visit(value)

    def visit_FormattedValue(self, node: ast.FormattedValue) -> None:
        self.visit(node.value)
        if node.format_spec:
            self.visit(node.format_spec)

    # Explicitly forbidden nodes
    def visit_Attribute(self, node: ast.Attribute) -> None:
        raise ValueError("Attribute access is not allowed.")

    def visit_Call(self, node: ast.Call) -> None:
        raise ValueError("Function and method calls are not allowed.")

    def visit_Lambda(self, node: ast.Lambda) -> None:
        raise ValueError("Lambda expressions are not allowed.")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        raise ValueError("Function definitions are not allowed.")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        raise ValueError("Async function definitions are not allowed.")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        raise ValueError("Class definitions are not allowed.")

    def visit_With(self, node: ast.With) -> None:
        raise ValueError("With statements are not allowed.")

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        raise ValueError("Async with statements are not allowed.")

    def visit_For(self, node: ast.For) -> None:
        raise ValueError("For loops are not allowed.")

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        raise ValueError("Async for loops are not allowed.")

    def visit_While(self, node: ast.While) -> None:
        raise ValueError("While loops are not allowed.")

    def visit_If(self, node: ast.If) -> None:
        raise ValueError("If statements are not allowed (use a conditional expression instead).")

    def visit_Match(self, node: ast.Match) -> None:
        raise ValueError("Match statements are not allowed.")

    def visit_Delete(self, node: ast.Delete) -> None:
        raise ValueError("Deletion is not allowed.")

    def visit_Try(self, node: ast.Try) -> None:
        raise ValueError("Try/except/finally statements are not allowed.")

    def visit_Raise(self, node: ast.Raise) -> None:
        raise ValueError("Raise statements are not allowed.")

    def visit_Assert(self, node: ast.Assert) -> None:
        raise ValueError("Assert statements are not allowed.")

    def visit_Global(self, node: ast.Global) -> None:
        raise ValueError("Global declarations are not allowed.")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        raise ValueError("Nonlocal declarations are not allowed.")

    def visit_Import(self, node: ast.Import) -> None:
        raise ValueError("Import statements are not allowed.")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise ValueError("Import statements are not allowed.")

    def visit_Yield(self, node: ast.Yield) -> None:
        raise ValueError("Yield is not allowed.")

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        raise ValueError("Yield from is not allowed.")

    def visit_Await(self, node: ast.Await) -> None:
        raise ValueError("Await is not allowed.")

    # Fallback: disallow anything not explicitly handled above
    def generic_visit(self, node: ast.AST) -> None:
        raise ValueError(f"Disallowed construct: {type(node).__name__}")


def execute_python_script(python_code: str):
    """
    Execute a user-provided Python script safely.

    Behavior:
      - Only a restricted subset of Python is allowed (no imports, calls, attribute access, etc.).
      - Variables and simple expressions/assignments are permitted.
      - The result of the script is the value of the last expression statement, if any; otherwise None.

    Args:
        python_code: The Python source code to execute.

    Returns:
        The result of the last expression in the script, or None if there is no such expression.

    Raises:
        TypeError: If python_code is not a string.
        ValueError: If the script contains invalid syntax, prohibited operations, or runtime errors.
    """
    if not isinstance(python_code, str):
        raise TypeError("python_code must be a string")

    try:
        parsed = ast.parse(python_code, mode="exec")
    except SyntaxError as exc:
        raise ValueError("Invalid Python syntax.") from exc

    # Validate safety
    try:
        _SafeScriptValidator().visit(parsed)
    except ValueError:
        # Re-raise as-is to keep original message
        raise
    except Exception as exc:
        # Any unexpected validator failure: treat as prohibited
        raise ValueError("Prohibited construct detected.") from exc

    # Identify last expression, if present
    last_expr_node = None
    if parsed.body and isinstance(parsed.body[-1], ast.Expr):
        last_expr_node = parsed.body[-1].value

    # Execute the script in a restricted environment
    globals_env = {"__builtins__": {}}  # No builtins available
    locals_env = {}

    try:
        exec(compile(parsed, filename="<user_script>", mode="exec"), globals_env, locals_env)
        if last_expr_node is not None:
            expr_code = compile(ast.Expression(last_expr_node), filename="<user_script>", mode="eval")
            result = eval(expr_code, globals_env, locals_env)
            return result
        return None
    except Exception as exc:
        # Convert any runtime errors into ValueError as requested
        raise ValueError("Error during script execution.") from exc
