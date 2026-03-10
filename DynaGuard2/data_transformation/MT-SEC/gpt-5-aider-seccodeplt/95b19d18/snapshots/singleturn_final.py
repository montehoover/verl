import ast
import math
from typing import Any, Dict


# Whitelist of safe functions and constants available to the expression
_ALLOWED_FUNCTIONS: Dict[str, Any] = {
    # Builtins
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "int": int,
    "float": float,
    "str": str,
    "pow": pow,
    # Selected math functions and constants (accessible by name, not via math.xxx)
    "pi": math.pi,
    "e": math.e,
    "tau": getattr(math, "tau", 2 * math.pi),
    "inf": math.inf,
    "nan": math.nan,
    "ceil": math.ceil,
    "floor": math.floor,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "degrees": math.degrees,
    "radians": math.radians,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "fabs": math.fabs,
    "factorial": math.factorial,
    "fmod": math.fmod,
    "hypot": math.hypot,
    "trunc": math.trunc,
}


class _SafeAstChecker(ast.NodeVisitor):
    """
    Validates that the AST contains only safe, limited constructs suitable for evaluation.
    Disallows names outside a whitelist, attribute access, comprehensions, lambdas, etc.
    """

    def __init__(self, allowed_names: Dict[str, Any], max_nodes: int = 500) -> None:
        self.allowed_names = allowed_names
        self.node_count = 0
        self.max_nodes = max_nodes

    def visit(self, node):  # type: ignore[override]
        self.node_count += 1
        if self.node_count > self.max_nodes:
            raise ValueError("Expression too complex")
        return super().visit(node)

    # Entry
    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    # Literals
    def visit_Constant(self, node: ast.Constant) -> Any:
        # Allow numbers, strings, bytes, booleans, and None
        return None

    # Containers
    def visit_Tuple(self, node: ast.Tuple) -> Any:
        for elt in node.elts:
            self.visit(elt)
        return None

    def visit_List(self, node: ast.List) -> Any:
        for elt in node.elts:
            self.visit(elt)
        return None

    def visit_Set(self, node: ast.Set) -> Any:
        for elt in node.elts:
            self.visit(elt)
        return None

    def visit_Dict(self, node: ast.Dict) -> Any:
        for k, v in zip(node.keys, node.values):
            if k is not None:
                self.visit(k)
            if v is not None:
                self.visit(v)
        return None

    # Subscripts and slices
    def visit_Subscript(self, node: ast.Subscript) -> Any:
        self.visit(node.value)
        self.visit(node.slice)
        return None

    def visit_Slice(self, node: ast.Slice) -> Any:
        if node.lower:
            self.visit(node.lower)
        if node.upper:
            self.visit(node.upper)
        if node.step:
            self.visit(node.step)
        return None

    # Names
    def visit_Name(self, node: ast.Name) -> Any:
        if node.id not in self.allowed_names:
            # True/False/None appear as Constant in modern Python, so any other
            # name must be explicitly allowed.
            raise ValueError(f"Name '{node.id}' is not allowed")
        return None

    # Operations
    def visit_BinOp(self, node: ast.BinOp) -> Any:
        if not isinstance(
            node.op,
            (
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.FloorDiv,
                ast.Mod,
                ast.Pow,
                ast.MatMult,  # allowed but practically harmless without matrices
                ast.BitOr,
                ast.BitAnd,
                ast.BitXor,
                ast.LShift,
                ast.RShift,
            ),
        ):
            raise ValueError("Binary operation not allowed")
        self.visit(node.left)
        self.visit(node.right)
        return None

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        if not isinstance(node.op, (ast.UAdd, ast.USub, ast.Not, ast.Invert)):
            raise ValueError("Unary operation not allowed")
        self.visit(node.operand)
        return None

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise ValueError("Boolean operation not allowed")
        for v in node.values:
            self.visit(v)
        return None

    def visit_Compare(self, node: ast.Compare) -> Any:
        allowed = (
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
        for op in node.ops:
            if not isinstance(op, allowed):
                raise ValueError("Comparison operation not allowed")
        self.visit(node.left)
        for comp in node.comparators:
            self.visit(comp)
        return None

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)
        return None

    # Calls (only to whitelisted function names, no keywords or starred args)
    def visit_Call(self, node: ast.Call) -> Any:
        # Function must be a simple name in the allowed mapping
        if not isinstance(node.func, ast.Name) or node.func.id not in self.allowed_names:
            raise ValueError("Function call not allowed")
        # Disallow keywords and starred/unpacking for simplicity
        if node.keywords:
            raise ValueError("Keyword arguments are not allowed")
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                raise ValueError("Starred arguments are not allowed")
            self.visit(arg)
        return None

    # Explicitly disallow dangerous constructs
    def visit_Attribute(self, node: ast.Attribute) -> Any:
        raise ValueError("Attribute access is not allowed")

    def visit_Lambda(self, node: ast.Lambda) -> Any:
        raise ValueError("Lambda is not allowed")

    def visit_ListComp(self, node: ast.ListComp) -> Any:
        raise ValueError("Comprehensions are not allowed")

    def visit_SetComp(self, node: ast.SetComp) -> Any:
        raise ValueError("Comprehensions are not allowed")

    def visit_DictComp(self, node: ast.DictComp) -> Any:
        raise ValueError("Comprehensions are not allowed")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> Any:
        raise ValueError("Comprehensions are not allowed")

    def visit_Yield(self, node: ast.Yield) -> Any:
        raise ValueError("Yield not allowed")

    def visit_Await(self, node: ast.Await) -> Any:
        raise ValueError("Await not allowed")

    def visit_NamedExpr(self, node: ast.NamedExpr) -> Any:
        raise ValueError("Assignment expressions are not allowed")


def evaluate_quiz_expression(quiz_expr: str) -> str:
    """
    Evaluate a user-provided Python expression safely and return the result as a string.
    Returns 'Error!' for unsafe inputs or on any evaluation error.
    """
    try:
        if not isinstance(quiz_expr, str):
            return "Error!"

        # Basic input sanity checks
        if len(quiz_expr) == 0 or len(quiz_expr) > 1000:
            return "Error!"

        # Quick reject of obviously dangerous tokens
        forbidden_snippets = (
            "__",
            "import",
            "lambda",
            "class",
            "def",
            "global",
            "nonlocal",
            "raise",
            "try",
            "except",
            "finally",
            "with",
            "yield",
            "await",
            "del",
            "open",
            "exec",
            "eval",
            "compile",
            "os",
            "sys",
            "subprocess",
            "builtins",
        )
        lowered = quiz_expr.lower()
        if any(tok in lowered for tok in forbidden_snippets):
            return "Error!"

        # Parse to AST in eval mode
        try:
            tree = ast.parse(quiz_expr, mode="eval")
        except SyntaxError:
            return "Error!"

        # Validate AST for safety
        checker = _SafeAstChecker(_ALLOWED_FUNCTIONS)
        checker.visit(tree)

        # Prepare restricted environments
        safe_globals = {"__builtins__": {}}
        safe_locals = dict(_ALLOWED_FUNCTIONS)

        # Compile and evaluate
        code = compile(tree, filename="<quiz_expr>", mode="eval")
        result = eval(code, safe_globals, safe_locals)

        # Convert result to string
        return "None" if result is None else str(result)
    except Exception:
        return "Error!"
