import ast

# Whitelisted, safe builtins that user code is allowed to call.
_ALLOWED_BUILTINS = {
    "abs": abs,
    "round": round,
    "len": len,
    "sum": sum,
    "min": min,
    "max": max,
    "sorted": sorted,
    "range": range,
    "print": print,
    "enumerate": enumerate,
    "zip": zip,
    "all": all,
    "any": any,
}


class _SafetyValidator(ast.NodeVisitor):
    """
    AST validator that rejects unsafe constructs and limits callable functions
    to a whitelist of safe builtins. Raises ValueError on violations.
    """

    def __init__(self, allowed_calls):
        super().__init__()
        self.allowed_calls = set(allowed_calls)

    # ---------- Helpers ----------

    def _error(self, node, msg):
        raise ValueError(f"Prohibited operation: {msg}")

    def _ensure_allowed_binop(self, node):
        # Disallow power and matrix multiplication, which can be expensive/abusive
        disallowed_ops = (ast.Pow, ast.MatMult)
        if isinstance(node.op, disallowed_ops):
            self._error(node, f"operator {type(node.op).__name__} is not allowed")

    def _ensure_safe_name(self, node: ast.Name):
        # Disallow dunder names and direct reference to __builtins__
        if node.id.startswith("__"):
            self._error(node, "access to dunder names is not allowed")
        if node.id == "__builtins__":
            self._error(node, "access to __builtins__ is not allowed")

    # ---------- Explicitly banned nodes ----------

    def visit_Import(self, node):
        self._error(node, "import statements are not allowed")

    def visit_ImportFrom(self, node):
        self._error(node, "import statements are not allowed")

    def visit_Attribute(self, node):
        # Prevent attribute traversal like obj.__class__ or os.system
        self._error(node, "attribute access is not allowed")

    def visit_Lambda(self, node):
        self._error(node, "lambda is not allowed")

    def visit_FunctionDef(self, node):
        self._error(node, "function definitions are not allowed")

    def visit_AsyncFunctionDef(self, node):
        self._error(node, "function definitions are not allowed")

    def visit_ClassDef(self, node):
        self._error(node, "class definitions are not allowed")

    def visit_With(self, node):
        self._error(node, "with statements are not allowed")

    def visit_AsyncWith(self, node):
        self._error(node, "with statements are not allowed")

    def visit_Try(self, node):
        self._error(node, "try/except/finally is not allowed")

    def visit_Raise(self, node):
        self._error(node, "raising exceptions is not allowed")

    def visit_Global(self, node):
        self._error(node, "global is not allowed")

    def visit_Nonlocal(self, node):
        self._error(node, "nonlocal is not allowed")

    def visit_Delete(self, node):
        self._error(node, "delete statements are not allowed")

    def visit_While(self, node):
        self._error(node, "while loops are not allowed")

    def visit_For(self, node):
        self._error(node, "for loops are not allowed")

    def visit_AsyncFor(self, node):
        self._error(node, "for loops are not allowed")

    def visit_ListComp(self, node):
        self._error(node, "comprehensions are not allowed")

    def visit_SetComp(self, node):
        self._error(node, "comprehensions are not allowed")

    def visit_DictComp(self, node):
        self._error(node, "comprehensions are not allowed")

    def visit_GeneratorExp(self, node):
        self._error(node, "comprehensions are not allowed")

    def visit_Yield(self, node):
        self._error(node, "yield is not allowed")

    def visit_YieldFrom(self, node):
        self._error(node, "yield is not allowed")

    def visit_Await(self, node):
        self._error(node, "await is not allowed")

    # ---------- Controlled nodes ----------

    def visit_Name(self, node: ast.Name):
        self._ensure_safe_name(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Only allow calls to whitelisted plain names (no attributes)
        if not isinstance(node.func, ast.Name):
            self._error(node, "only calls to whitelisted functions are allowed")
        func_name = node.func.id
        if func_name not in self.allowed_calls:
            self._error(node, f"call to '{func_name}' is not allowed")
        # Validate arguments as normal
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp):
        self._ensure_allowed_binop(node)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        # Ensure the operator used in augmented assignment is allowed
        fake_bin = ast.BinOp(left=node.target, op=node.op, right=node.value)
        self._ensure_allowed_binop(fake_bin)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        # Disallow assignment to attributes; allow names, tuples, lists, and subscripts
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                self._error(node, "assignment to attributes is not allowed")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        # Allow subscripting but attribute access is already banned elsewhere
        self.generic_visit(node)

    def visit_If(self, node: ast.If):
        # Safe enough in terms of capability; allow simple conditionals
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr):
        self.generic_visit(node)

    # Constants, containers, comparisons, f-strings, etc. are handled via generic_visit


def execute_python_script(python_code):
    """
    Execute a user-provided Python script safely.

    Args:
        python_code (str): The Python code to execute.

    Returns:
        The result of the executed script if it ends with an expression, otherwise None.

    Raises:
        ValueError: If the script contains invalid syntax or prohibited operations.
        Other exceptions raised during execution will propagate.
    """
    if not isinstance(python_code, str):
        raise ValueError("python_code must be a string")

    try:
        tree = ast.parse(python_code, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax: {e.msg}") from None

    # Validate AST for safety
    _SafetyValidator(_ALLOWED_BUILTINS.keys()).visit(tree)

    # Create a restricted execution environment with only whitelisted builtins
    safe_globals = {"__builtins__": {}}
    safe_globals.update(_ALLOWED_BUILTINS)
    safe_locals = safe_globals  # Use the same dict for globals/locals

    # If the script ends with a bare expression, return its value
    if len(tree.body) > 0 and isinstance(tree.body[-1], ast.Expr):
        pre_body = tree.body[:-1]
        last_expr = tree.body[-1].value

        if pre_body:
            pre_module = ast.Module(body=pre_body, type_ignores=[])
            code_exec = compile(pre_module, filename="<user_script>", mode="exec")
            exec(code_exec, safe_globals, safe_locals)

        expr_obj = ast.Expression(last_expr)
        code_eval = compile(expr_obj, filename="<user_script>", mode="eval")
        return eval(code_eval, safe_globals, safe_locals)

    # Otherwise, execute statements and return None
    code = compile(tree, filename="<user_script>", mode="exec")
    exec(code, safe_globals, safe_locals)
    return None
