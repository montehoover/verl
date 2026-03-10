import ast
from typing import Any


class _UnsafeOperation(Exception):
    pass


class _SafeEvalValidator(ast.NodeVisitor):
    # Limits to mitigate resource abuse
    MAX_INT_ABS = 10**9
    MAX_FLOAT_ABS = 1e12
    MAX_STR_LEN = 10000
    MAX_COLLECTION_SIZE = 1000

    # Allowed operations
    ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)
    ALLOWED_UNARYOPS = (ast.UAdd, ast.USub, ast.Not)
    ALLOWED_BOOLOPS = (ast.And, ast.Or)
    ALLOWED_CMPOPS = (
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.NotIn,
        ast.Is,
        ast.IsNot,
    )

    def visit(self, node: ast.AST) -> Any:
        # Disallow any node types that aren't explicitly handled below
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, None)
        if visitor is None:
            raise _UnsafeOperation(f"Disallowed node type: {node.__class__.__name__}")
        return visitor(node)

    def visit_Expression(self, node: ast.Expression) -> None:
        self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> None:
        v = node.value
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, int):
            if abs(v) > self.MAX_INT_ABS:
                raise _UnsafeOperation("Integer constant too large")
            return
        if isinstance(v, float):
            if not (abs(v) <= self.MAX_FLOAT_ABS):
                raise _UnsafeOperation("Float constant too large")
            return
        if isinstance(v, (str, bytes)):
            if len(v) > self.MAX_STR_LEN:
                raise _UnsafeOperation("String/bytes literal too long")
            return
        # Disallow other constant types
        raise _UnsafeOperation("Unsupported constant type")

    # For Python < 3.8 True/False/None may appear as Name nodes
    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in ("True", "False", "None"):
            raise _UnsafeOperation("Names are not allowed")

    def visit_Tuple(self, node: ast.Tuple) -> None:
        if len(node.elts) > self.MAX_COLLECTION_SIZE:
            raise _UnsafeOperation("Tuple too large")
        for e in node.elts:
            self.visit(e)

    def visit_List(self, node: ast.List) -> None:
        if len(node.elts) > self.MAX_COLLECTION_SIZE:
            raise _UnsafeOperation("List too large")
        for e in node.elts:
            self.visit(e)

    def visit_Set(self, node: ast.Set) -> None:
        if len(node.elts) > self.MAX_COLLECTION_SIZE:
            raise _UnsafeOperation("Set too large")
        for e in node.elts:
            self.visit(e)

    def visit_Dict(self, node: ast.Dict) -> None:
        if len(node.keys or []) > self.MAX_COLLECTION_SIZE:
            raise _UnsafeOperation("Dict too large")
        for k, v in zip(node.keys or [], node.values or []):
            if k is not None:
                self.visit(k)
            self.visit(v)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, self.ALLOWED_BINOPS):
            raise _UnsafeOperation("Operator not allowed")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, self.ALLOWED_UNARYOPS):
            raise _UnsafeOperation("Unary operator not allowed")
        self.visit(node.operand)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if not isinstance(node.op, self.ALLOWED_BOOLOPS):
            raise _UnsafeOperation("Boolean operator not allowed")
        for v in node.values:
            self.visit(v)

    def visit_Compare(self, node: ast.Compare) -> None:
        self.visit(node.left)
        for op in node.ops:
            if not isinstance(op, self.ALLOWED_CMPOPS):
                raise _UnsafeOperation("Comparison operator not allowed")
        for comp in node.comparators:
            self.visit(comp)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self.visit(node.value)
        # slice can be Slice, Tuple, or expression
        self.visit(node.slice)

    def visit_Slice(self, node: ast.Slice) -> None:
        if node.lower:
            self.visit(node.lower)
        if node.upper:
            self.visit(node.upper)
        if node.step:
            self.visit(node.step)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        for v in node.values:
            self.visit(v)

    def visit_FormattedValue(self, node: ast.FormattedValue) -> None:
        # Only allow simple formatting of allowed expressions
        self.visit(node.value)
        if node.format_spec:
            self.visit(node.format_spec)

    # Explicitly disallow potentially dangerous constructs
    def visit_Call(self, node: ast.Call) -> None:
        raise _UnsafeOperation("Function calls are not allowed")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        raise _UnsafeOperation("Attribute access is not allowed")

    def visit_Lambda(self, node: ast.Lambda) -> None:
        raise _UnsafeOperation("Lambdas are not allowed")

    def visit_DictComp(self, node: ast.DictComp) -> None:
        raise _UnsafeOperation("Comprehensions are not allowed")

    def visit_SetComp(self, node: ast.SetComp) -> None:
        raise _UnsafeOperation("Comprehensions are not allowed")

    def visit_ListComp(self, node: ast.ListComp) -> None:
        raise _UnsafeOperation("Comprehensions are not allowed")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        raise _UnsafeOperation("Generators are not allowed")

    def visit_Yield(self, node: ast.Yield) -> None:
        raise _UnsafeOperation("Yield is not allowed")

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        raise _UnsafeOperation("Yield is not allowed")

    def visit_Await(self, node: ast.Await) -> None:
        raise _UnsafeOperation("Await is not allowed")

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        raise _UnsafeOperation("Assignment expressions are not allowed")


def execute_user_code(script_code: str) -> str:
    """
    Securely evaluates a small user-supplied Python expression limited to
    basic arithmetic and string manipulations.

    Args:
        script_code: str - A user-supplied Python expression.

    Returns:
        str - The result of evaluating the script as a string, or
              'Execution Blocked!' if the script is unsafe or invalid.
    """
    if not isinstance(script_code, str):
        return "Execution Blocked!"

    try:
        # Parse only as an expression to disallow statements
        parsed = ast.parse(script_code, mode="eval")
    except Exception:
        return "Execution Blocked!"

    # Validate AST against a strict whitelist
    try:
        _SafeEvalValidator().visit(parsed)
    except _UnsafeOperation:
        return "Execution Blocked!"
    except Exception:
        return "Execution Blocked!"

    # Evaluate with no builtins or globals
    try:
        result = eval(
            compile(parsed, filename="<user>", mode="eval"),
            {"__builtins__": {}},
            {},
        )
    except Exception:
        return "Execution Blocked!"

    # Convert result to string safely
    try:
        s = str(result)
    except Exception:
        return "Execution Blocked!"

    # Optional: enforce a reasonable output size
    if len(s) > _SafeEvalValidator.MAX_STR_LEN:
        return "Execution Blocked!"

    return s
