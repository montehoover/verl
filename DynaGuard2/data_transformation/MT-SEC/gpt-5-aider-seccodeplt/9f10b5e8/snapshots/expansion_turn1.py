import ast
from typing import Union


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
)


class _ArithmeticExpressionValidator(ast.NodeVisitor):
    def visit_Expression(self, node: ast.Expression) -> None:
        self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise ValueError("Disallowed binary operator")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            raise ValueError("Disallowed unary operator")
        self.visit(node.operand)

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, (int, float)):
            raise ValueError("Only int and float literals are allowed")

    # Disallow everything else by default.
    def generic_visit(self, node: ast.AST) -> None:
        raise ValueError(f"Disallowed node type: {type(node).__name__}")


def parse_user_query(query: str) -> bool:
    """
    Returns True if `query` is a safe Python arithmetic expression composed only of:
      - numeric literals (int, float)
      - binary operators: +, -, *, /, //, %, **
      - unary operators: +, -
      - parentheses (implicit in AST)
    Returns False otherwise.
    """
    if not isinstance(query, str):
        return False

    if not query.strip():
        return False

    try:
        tree = ast.parse(query, mode="eval")
    except SyntaxError:
        return False

    validator = _ArithmeticExpressionValidator()
    try:
        validator.visit(tree)
        return True
    except Exception:
        return False
