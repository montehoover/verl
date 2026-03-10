import re
import ast

_ALLOWED_CHARS_RE = re.compile(r'^[\s0-9+\-*/%()._]+$')


def _safe_div(a, b):
    if b == 0:
        raise ValueError("Division by zero.")
    return a / b


def _safe_floordiv(a, b):
    if b == 0:
        raise ValueError("Division by zero.")
    return a // b


def _safe_mod(a, b):
    if b == 0:
        raise ValueError("Modulo by zero.")
    return a % b


_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: _safe_div,
    ast.FloorDiv: _safe_floordiv,
    ast.Mod: _safe_mod,
    ast.Pow: lambda a, b: a ** b,
}

_UNARY_OPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


class _SafeEvaluator(ast.NodeVisitor):
    def generic_visit(self, node):
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)

        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            raise ValueError("Invalid operands in expression.")

        op_func = _BIN_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")

        result = op_func(left, right)
        if isinstance(result, complex):
            raise ValueError("Complex numbers are not supported.")
        return result

    def visit_UnaryOp(self, node: ast.UnaryOp):
        operand = self.visit(node.operand)
        if not isinstance(operand, (int, float)):
            raise ValueError("Invalid operand in unary operation.")

        op_func = _UNARY_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

        return op_func(operand)

    # Python <3.8 numeric literal
    def visit_Num(self, node: ast.Num):
        value = node.n
        if isinstance(value, bool):
            raise ValueError("Booleans are not supported.")
        if not isinstance(value, (int, float)):
            raise ValueError(f"Unsupported numeric literal: {type(value).__name__}")
        return value

    # Python 3.8+ constant
    def visit_Constant(self, node: ast.Constant):
        value = node.value
        if isinstance(value, bool):
            raise ValueError("Booleans are not supported.")
        if not isinstance(value, (int, float)):
            raise ValueError(f"Unsupported constant: {type(value).__name__}")
        return value


def evaluate_expression(expr: str):
    """
    Evaluate a simple mathematical expression safely.

    Args:
        expr: str - a string representing the arithmetic expression.

    Returns:
        The computed result of the expression (int or float).

    Raises:
        ValueError: if unsupported characters, unsafe commands, or invalid
                    operations are detected (e.g., division by zero).
    """
    if not isinstance(expr, str):
        raise ValueError("Expression must be a string.")

    expr = expr.strip()
    if not expr:
        raise ValueError("Empty expression.")

    # Quick unsupported-character check (only digits, whitespace, parentheses, and basic operators)
    if not _ALLOWED_CHARS_RE.fullmatch(expr):
        raise ValueError("Unsupported characters detected in expression.")

    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise ValueError("Invalid expression.")

    evaluator = _SafeEvaluator()
    result = evaluator.visit(parsed)

    # Normalize negative zero to positive zero for consistency
    if isinstance(result, float) and result == 0.0:
        return 0.0

    return result
