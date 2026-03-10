import ast


class _SafetyChecker(ast.NodeVisitor):
    """
    Validates that the AST only contains a safe subset of Python:
      - Statements: Assign, AugAssign, Expr
      - Expressions: Name, Constant, BinOp, UnaryOp, BoolOp, Compare,
                     IfExp, Tuple, List, Set, Dict
      - Disallows: calls, attribute access, subscripts, imports, control flow, etc.
    """

    _ALLOWED_BIN_OPS = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
    )

    _ALLOWED_UNARY_OPS = (
        ast.UAdd,
        ast.USub,
        ast.Not,
    )

    _ALLOWED_BOOL_OPS = (
        ast.And,
        ast.Or,
    )

    _ALLOWED_CMP_OPS = (
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

    _ALLOWED_CONSTANT_TYPES = (int, float, str, bool, type(None))

    def error(self, node, msg="Disallowed operation"):
        raise ValueError(msg)

    # Module -> sequence of statements
    def visit_Module(self, node: ast.Module):
        for stmt in node.body:
            self.visit(stmt)

    # Allowed statements
    def visit_Expr(self, node: ast.Expr):
        self.visit(node.value)

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            self.error(node, "Only simple assignments to a single variable are allowed")
        self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign):
        if not isinstance(node.target, ast.Name):
            self.error(node, "Only augmented assignments to a variable are allowed")
        if not isinstance(node.op, self._ALLOWED_BIN_OPS):
            self.error(node, f"Operator not allowed in augmented assignment: {type(node.op).__name__}")
        self.visit(node.value)

    # Allowed expressions
    def visit_Name(self, node: ast.Name):
        # Variable names are allowed
        return

    def visit_Constant(self, node: ast.Constant):
        if not isinstance(node.value, self._ALLOWED_CONSTANT_TYPES):
            self.error(node, "Constant type not allowed")

    def visit_Tuple(self, node: ast.Tuple):
        for elt in node.elts:
            self.visit(elt)

    def visit_List(self, node: ast.List):
        for elt in node.elts:
            self.visit(elt)

    def visit_Set(self, node: ast.Set):
        for elt in node.elts:
            self.visit(elt)

    def visit_Dict(self, node: ast.Dict):
        for k, v in zip(node.keys, node.values):
            if k is not None:
                self.visit(k)
            self.visit(v)

    def visit_BinOp(self, node: ast.BinOp):
        if not isinstance(node.op, self._ALLOWED_BIN_OPS):
            self.error(node, f"Binary operator not allowed: {type(node.op).__name__}")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if not isinstance(node.op, self._ALLOWED_UNARY_OPS):
            self.error(node, f"Unary operator not allowed: {type(node.op).__name__}")
        self.visit(node.operand)

    def visit_BoolOp(self, node: ast.BoolOp):
        if not isinstance(node.op, self._ALLOWED_BOOL_OPS):
            self.error(node, f"Boolean operator not allowed: {type(node.op).__name__}")
        for v in node.values:
            self.visit(v)

    def visit_Compare(self, node: ast.Compare):
        for op in node.ops:
            if not isinstance(op, self._ALLOWED_CMP_OPS):
                self.error(node, f"Comparison operator not allowed: {type(op).__name__}")
        self.visit(node.left)
        for comp in node.comparators:
            self.visit(comp)

    def visit_IfExp(self, node: ast.IfExp):
        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)

    # Explicitly disallow potentially dangerous or complex constructs
    def visit_Call(self, node: ast.Call):
        self.error(node, "Function calls are not allowed")

    def visit_Attribute(self, node: ast.Attribute):
        self.error(node, "Attribute access is not allowed")

    def visit_Subscript(self, node: ast.Subscript):
        self.error(node, "Indexing/slicing is not allowed")

    def visit_ListComp(self, node: ast.ListComp):
        self.error(node, "Comprehensions are not allowed")

    def visit_SetComp(self, node: ast.SetComp):
        self.error(node, "Comprehensions are not allowed")

    def visit_DictComp(self, node: ast.DictComp):
        self.error(node, "Comprehensions are not allowed")

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        self.error(node, "Comprehensions are not allowed")

    def visit_Lambda(self, node: ast.Lambda):
        self.error(node, "Lambdas are not allowed")

    def visit_NamedExpr(self, node: ast.NamedExpr):
        self.error(node, "Assignment expressions are not allowed")

    def visit_Assert(self, node: ast.Assert):
        self.error(node)

    def visit_Await(self, node: ast.Await):
        self.error(node)

    def visit_Yield(self, node: ast.Yield):
        self.error(node)

    def visit_YieldFrom(self, node: ast.YieldFrom):
        self.error(node)

    def visit_Import(self, node: ast.Import):
        self.error(node, "Imports are not allowed")

    def visit_ImportFrom(self, node: ast.ImportFrom):
        self.error(node, "Imports are not allowed")

    def visit_Raise(self, node: ast.Raise):
        self.error(node)

    def visit_Try(self, node: ast.Try):
        self.error(node)

    def visit_With(self, node: ast.With):
        self.error(node)

    def visit_If(self, node: ast.If):
        self.error(node)

    def visit_For(self, node: ast.For):
        self.error(node)

    def visit_While(self, node: ast.While):
        self.error(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.error(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.error(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        self.error(node)

    def visit_Global(self, node: ast.Global):
        self.error(node)

    def visit_Nonlocal(self, node: ast.Nonlocal):
        self.error(node)

    def generic_visit(self, node):
        # Any node not explicitly allowed should be rejected
        disallowed = (
            ast.comprehension,
            ast.JoinedStr,
            ast.FormattedValue,
            ast.Starred,
            ast.Delete,
            ast.Slice,
            ast.ExtSlice,
        )
        if isinstance(node, disallowed):
            self.error(node)
        super().generic_visit(node)


def run_code_snippet(code: str):
    """
    Execute a small, safe Python code snippet.

    Rules:
      - Only simple assignments and expressions are allowed.
      - No function calls, attribute access, indexing, imports, or control flow.
      - Returns the value of the last expression statement, or None otherwise.
      - Raises ValueError if disallowed syntax/operations are used.

    Parameters:
        code (str): The code snippet to execute.

    Returns:
        Any: Result of the last expression, or None if there is no expression result.
    """
    if not isinstance(code, str):
        raise TypeError("code must be a string")

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Invalid code: {e}") from e

    # Validate AST for safety
    _SafetyChecker().visit(tree)

    # Prepare a restricted execution environment (no builtins)
    env = {"__builtins__": {}}

    # Execute the code
    compiled = compile(tree, filename="<snippet>", mode="exec")
    exec(compiled, env, env)

    # Determine and return result: value of the last expression statement, if any
    if not isinstance(tree, ast.Module) or not tree.body:
        return None

    last_stmt = tree.body[-1]
    if isinstance(last_stmt, ast.Expr):
        expr_code = compile(ast.Expression(last_stmt.value), filename="<snippet>", mode="eval")
        return eval(expr_code, env, env)

    return None
