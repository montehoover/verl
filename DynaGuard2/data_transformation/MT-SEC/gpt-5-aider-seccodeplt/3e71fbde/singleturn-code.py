import ast

# Safe builtins that are allowed to be used by the executed snippet.
_SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "range": range,
    "print": print,
    "sorted": sorted,
    "enumerate": enumerate,
    "all": all,
    "any": any,
    "round": round,
    "bool": bool,
    "int": int,
    "float": float,
    "str": str,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
}

_ALLOWED_CALLS = set(_SAFE_BUILTINS.keys())

# Whitelist of AST node types allowed in the snippet.
_ALLOWED_AST_NODES = {
    # Module/statement containers
    ast.Module,
    ast.Expr,
    ast.Assign,
    ast.AugAssign,

    # Names and constants
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,

    # Data structures
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Dict,

    # Indexing/slicing
    ast.Subscript,
    ast.Slice,

    # Expressions/operations
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,  # ternary: a if cond else b
    ast.Call,
    ast.keyword,

    # f-strings
    ast.JoinedStr,
    ast.FormattedValue,

    # Operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.BitOr,
    ast.BitAnd,
    ast.BitXor,
    ast.LShift,
    ast.RShift,
    ast.Invert,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.And,
    ast.Or,

    # Comparators
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
}


# Names that must not be read from the environment.
_DISALLOWED_NAME_LOADS = {
    "__builtins__",
    "__import__",
    "eval",
    "exec",
    "open",
    "compile",
    "input",
    "help",
    "quit",
    "exit",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "type",
}


class _SafetyChecker(ast.NodeVisitor):
    """AST validator that enforces a strict subset of Python for safety."""

    def generic_visit(self, node):
        # Enforce whitelist of node types.
        if type(node) not in _ALLOWED_AST_NODES:
            raise ValueError(f"Disallowed operation: {type(node).__name__}")
        super().generic_visit(node)

    # Explicitly disallow a range of risky constructs.
    def visit_Import(self, node):
        raise ValueError("Disallowed operation: import statements are not allowed")

    def visit_ImportFrom(self, node):
        raise ValueError("Disallowed operation: import statements are not allowed")

    def visit_Attribute(self, node):
        # Prevent access to obj.attr (including dunder attributes)
        raise ValueError("Disallowed operation: attribute access is not allowed")

    def visit_With(self, node):
        raise ValueError("Disallowed operation: with statements are not allowed")

    def visit_AsyncWith(self, node):
        raise ValueError("Disallowed operation: async with statements are not allowed")

    def visit_For(self, node):
        raise ValueError("Disallowed operation: for loops are not allowed")

    def visit_While(self, node):
        raise ValueError("Disallowed operation: while loops are not allowed")

    def visit_If(self, node):
        raise ValueError("Disallowed operation: if statements are not allowed")

    def visit_FunctionDef(self, node):
        raise ValueError("Disallowed operation: function definitions are not allowed")

    def visit_AsyncFunctionDef(self, node):
        raise ValueError("Disallowed operation: async function definitions are not allowed")

    def visit_ClassDef(self, node):
        raise ValueError("Disallowed operation: class definitions are not allowed")

    def visit_Lambda(self, node):
        raise ValueError("Disallowed operation: lambda expressions are not allowed")

    def visit_Try(self, node):
        raise ValueError("Disallowed operation: try/except is not allowed")

    def visit_Raise(self, node):
        raise ValueError("Disallowed operation: raising exceptions is not allowed")

    def visit_Return(self, node):
        raise ValueError("Disallowed operation: return statements are not allowed")

    def visit_Delete(self, node):
        raise ValueError("Disallowed operation: delete statements are not allowed")

    def visit_Global(self, node):
        raise ValueError("Disallowed operation: global statements are not allowed")

    def visit_Nonlocal(self, node):
        raise ValueError("Disallowed operation: nonlocal statements are not allowed")

    def visit_Yield(self, node):
        raise ValueError("Disallowed operation: yield is not allowed")

    def visit_YieldFrom(self, node):
        raise ValueError("Disallowed operation: yield from is not allowed")

    def visit_Await(self, node):
        raise ValueError("Disallowed operation: await is not allowed")

    def visit_Call(self, node: ast.Call):
        # Only allow calling a permitted builtin by name (no attribute or subscript calls).
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name not in _ALLOWED_CALLS:
                raise ValueError(f"Disallowed operation: calling '{func_name}' is not allowed")
        else:
            # e.g., obj.method(), getattr(...)(...), (lambda: ...)()
            raise ValueError("Disallowed operation: calling non-builtin or attribute functions is not allowed")
        # Continue validating arguments/keywords.
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        # Only allow simple name assignments (e.g., x = 1)
        for tgt in node.targets:
            if not isinstance(tgt, ast.Name):
                raise ValueError("Disallowed operation: assignments must target simple variables")
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        if not isinstance(node.target, ast.Name):
            raise ValueError("Disallowed operation: augmented assignments must target simple variables")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        # Disallow reading certain sensitive names and any dunder identifiers
        if isinstance(node.ctx, ast.Load):
            if node.id in _DISALLOWED_NAME_LOADS or node.id.startswith("__"):
                raise ValueError(f"Disallowed operation: accessing '{node.id}' is not allowed")
        self.generic_visit(node)


def run_code_snippet(snippet: str):
    """
    Execute a small Python code snippet safely.

    Args:
        snippet: str - Python code to execute.

    Returns:
        - The value of the final expression if the snippet ends with an expression.
        - None if there is no resulting value (e.g., only statements/assignments).

    Raises:
        ValueError: if the snippet attempts any disallowed operations.
        SyntaxError or runtime exceptions: for invalid code or errors during execution.
    """
    if not isinstance(snippet, str):
        raise TypeError("snippet must be a string")

    code = snippet.strip()
    if code == "":
        return None

    # Parse and validate the AST.
    tree = ast.parse(code, mode="exec")
    _SafetyChecker().visit(tree)

    # Prepare a restricted global/local environment.
    env = {"__builtins__": dict(_SAFE_BUILTINS)}

    # If the last statement is an expression, evaluate it and return its value.
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        prefix_stmts = tree.body[:-1]
        if prefix_stmts:
            prefix_module = ast.Module(body=prefix_stmts, type_ignores=[])
            exec(compile(prefix_module, "<snippet>", "exec"), env, env)

        last_expr = tree.body[-1].value
        expr_code = compile(ast.Expression(last_expr), "<snippet>", "eval")
        return eval(expr_code, env, env)

    # Otherwise, just execute and return None.
    exec(compile(tree, "<snippet>", "exec"), env, env)
    return None
