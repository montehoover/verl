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


class _SafeArithmeticEvaluator(ast.NodeVisitor):
    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant):
        val = node.value
        if not isinstance(val, (int, float)):
            raise ValueError("Non-numeric literal")
        return val

    def visit_UnaryOp(self, node: ast.UnaryOp):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError("Disallowed unary operator")

    def visit_BinOp(self, node: ast.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise ZeroDivisionError("floor division by zero")
            return left // right
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise ZeroDivisionError("modulo by zero")
            return left % right
        if isinstance(node.op, ast.Pow):
            return left ** right

        raise ValueError("Disallowed binary operator")

    # Disallow anything else
    def generic_visit(self, node: ast.AST):
        raise ValueError(f"Disallowed node type: {type(node).__name__}")


def evaluate_safe_query(query: str) -> str:
    """
    Evaluate a validated arithmetic query safely.
    Returns the result as a string, or an error message indicating a potential safety violation.
    """
    error_msg = "Error: potential safety violation"

    if not isinstance(query, str) or not query.strip():
        return error_msg

    try:
        tree = ast.parse(query, mode="eval")
    except SyntaxError:
        return error_msg

    # Re-validate AST for safety before evaluation
    try:
        _ArithmeticExpressionValidator().visit(tree)
    except Exception:
        return error_msg

    # Evaluate using a strict evaluator that only supports allowed nodes/ops
    try:
        result = _SafeArithmeticEvaluator().visit(tree)
    except Exception:
        return error_msg

    if not isinstance(result, (int, float)):
        return error_msg

    return str(result)
