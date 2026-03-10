import ast
from typing import Any


__all__ = ["validate_expression"]


class _SafeExpressionValidator(ast.NodeVisitor):
    """
    Validates that an expression AST only contains safe arithmetic:
    - Literals: int, float (no bool, None, complex)
    - Operators: +, -, *, /, //, %, **
    - Unary operators: +, -
    - Parentheses are naturally allowed by the AST structure
    Disallows: names, calls, attribute access, indexing, bitwise ops,
    comparisons, boolean ops, lambdas, comprehensions, etc.
    """

    allowed_bin_ops = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
    )

    allowed_unary_ops = (
        ast.UAdd,
        ast.USub,
    )

    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,  # Python 3.8+: numbers
        # Note: ast.Load may appear as a context; we allow it via special-case
    )

    def generic_visit(self, node: ast.AST) -> Any:
        # Only traverse into explicitly allowed node types
        if isinstance(node, self.allowed_nodes):
            return super().generic_visit(node)
        # Contexts like ast.Load can appear but are safe
        if isinstance(node, ast.Load):
            return
        # Anything else is disallowed
        raise ValueError(f"Disallowed node: {node.__class__.__name__}")

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        if not isinstance(node.op, self.allowed_bin_ops):
            raise ValueError(f"Disallowed binary operator: {node.op.__class__.__name__}")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        if not isinstance(node.op, self.allowed_unary_ops):
            raise ValueError(f"Disallowed unary operator: {node.op.__class__.__name__}")
        self.visit(node.operand)

    def visit_Constant(self, node: ast.Constant) -> Any:
        # Only allow int/float literals. Reject bool (subclass of int), None, complex, strings, etc.
        v = node.value
        if type(v) not in (int, float):  # use type(...) to ensure bool is rejected
            raise ValueError(f"Disallowed literal type: {type(v).__name__}")


def validate_expression(expr: str) -> bool:
    """
    Returns True if the given expression string is a safe basic arithmetic expression,
    otherwise False.
    Allowed:
      - integers and floats (including scientific notation)
      - operators: +, -, *, /, //, %, **
      - unary + and -
      - parentheses and whitespace
    Disallowed:
      - names, function calls, attributes, indexing, bitwise ops, comparisons, etc.
      - non-numeric literals (strings, bytes, bools, None, complex)
    """
    if not isinstance(expr, str):
        return False

    # Basic sanity checks to prevent pathological inputs
    s = expr.strip()
    if not s:
        return False
    if len(s) > 1000:
        return False

    try:
        # Parse in 'eval' mode to only allow expressions, not statements
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        return False

    validator = _SafeExpressionValidator()
    try:
        validator.visit(tree)
    except ValueError:
        return False

    return True
