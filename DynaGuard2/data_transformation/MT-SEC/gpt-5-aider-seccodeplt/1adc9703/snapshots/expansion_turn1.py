import ast
import math
from typing import Any


class _SafeExprValidator(ast.NodeVisitor):
    """
    Validates that an expression AST contains only basic arithmetic:
    - numbers (int, float)
    - binary ops: +, -, *, /, %, //
    - unary ops: +, -
    - parentheses (implicit via AST)
    And nothing else.
    Also applies a node-count limit to avoid overly complex expressions.
    """

    def __init__(self, max_nodes: int = 1000) -> None:
        self._visited_nodes = 0
        self._max_nodes = max_nodes

    def visit(self, node: ast.AST) -> Any:
        self._visited_nodes += 1
        if self._visited_nodes > self._max_nodes:
            raise ValueError("Expression too complex")
        return super().visit(node)

    # Root
    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    # Allowed constants: int and float (finite), reject bool and others
    def visit_Constant(self, node: ast.Constant) -> Any:
        value = node.value
        if type(value) is int:
            return None
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("Non-finite float not allowed")
            return None
        raise ValueError("Only int and float literals are allowed")

    # Backward compatibility (older Python ASTs)
    def visit_Num(self, node: ast.Num) -> Any:  # type: ignore[attr-defined]
        value = node.n  # type: ignore[attr-defined]
        if type(value) is int:
            return None
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("Non-finite float not allowed")
            return None
        raise ValueError("Only int and float literals are allowed")

    # Binary operations: +, -, *, /, %, //
    def visit_BinOp(self, node: ast.BinOp) -> Any:
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.FloorDiv)):
            raise ValueError("Only basic arithmetic operators are allowed")
        self.visit(node.left)
        self.visit(node.right)
        return None

    # Unary operations: +, -
    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise ValueError("Only unary plus/minus are allowed")
        self.visit(node.operand)
        return None

    # Disallow everything else explicitly
    def visit_Name(self, node: ast.Name) -> Any:  # noqa: N802
        raise ValueError("Names are not allowed")

    def visit_Attribute(self, node: ast.Attribute) -> Any:  # noqa: N802
        raise ValueError("Attributes are not allowed")

    def visit_Call(self, node: ast.Call) -> Any:  # noqa: N802
        raise ValueError("Function calls are not allowed")

    def visit_Subscript(self, node: ast.Subscript) -> Any:  # noqa: N802
        raise ValueError("Subscripts are not allowed")

    def visit_List(self, node: ast.List) -> Any:  # noqa: N802
        raise ValueError("Lists are not allowed")

    def visit_Tuple(self, node: ast.Tuple) -> Any:  # noqa: N802
        raise ValueError("Tuples are not allowed")

    def generic_visit(self, node: ast.AST) -> Any:
        # If we haven't provided a specific visit_ method for the node type,
        # it is disallowed by default.
        disallowed = (
            ast.Dict, ast.Set, ast.Compare, ast.BoolOp, ast.IfExp, ast.Lambda,
            ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
            ast.Await, ast.Yield, ast.YieldFrom, ast.Assign, ast.AugAssign,
            ast.AnnAssign, ast.For, ast.While, ast.If, ast.With, ast.Try,
            ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Pass,
            ast.Break, ast.Continue, ast.Return, ast.Raise, ast.FormattedValue,
            ast.JoinedStr, ast.Bytes, ast.Str, ast.Slice, ast.ExtSlice,
            ast.Starred, ast.Match, ast.NamedExpr,
        )
        if isinstance(node, disallowed):
            raise ValueError(f"Disallowed syntax: {type(node).__name__}")
        return super().generic_visit(node)


def parse_expression(expression: str) -> bool:
    """
    Validate whether the given string is a safe basic arithmetic expression.

    Rules:
    - Only integers and finite floats.
    - Operators: +, -, *, /, %, // and parentheses.
    - Unary + and - are allowed.
    - No names, calls, attributes, subscripts, comprehensions, conditionals, etc.
    - Bounded complexity (node count) to avoid pathological inputs.

    Returns:
        True if the expression is valid and safe, False otherwise.
    """
    if not isinstance(expression, str):
        return False

    # Quick reject of empty/whitespace-only input
    if expression.strip() == "":
        return False

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return False

    try:
        validator = _SafeExprValidator(max_nodes=1000)
        validator.visit(tree)
        return True
    except Exception:
        return False
