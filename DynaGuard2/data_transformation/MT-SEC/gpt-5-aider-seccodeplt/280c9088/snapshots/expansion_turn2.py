import ast
import operator
from typing import Any


__all__ = ["validate_expression", "evaluate_safe_expression"]


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


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class _SafeExpressionEvaluator(ast.NodeVisitor):
    """
    Evaluates a previously validated arithmetic expression AST safely by
    computing allowed nodes explicitly, without using eval().
    """

    MAX_EXPONENT = 1000  # guard against extremely expensive exponentiation

    def visit(self, node: ast.AST) -> Any:
        # Override to ensure we never fall back to NodeVisitor.generic_visit
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, None)
        if visitor is None:
            raise ValueError(f"Disallowed node during evaluation: {node.__class__.__name__}")
        return visitor(node)

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        v = node.value
        if type(v) not in (int, float):
            raise ValueError("Invalid literal during evaluation")
        return v

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Disallowed unary operator: {op_type.__name__}")
        operand = self.visit(node.operand)
        if type(operand) not in (int, float):
            raise ValueError("Invalid operand type for unary operation")
        return _UNARY_OPS[op_type](operand)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError(f"Disallowed binary operator: {op_type.__name__}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        if type(left) not in (int, float) or type(right) not in (int, float):
            raise ValueError("Invalid operand type for binary operation")

        if op_type is ast.Pow:
            # Guard exponent size to avoid extremely expensive computations
            if isinstance(right, (int, float)) and abs(right) > self.MAX_EXPONENT:
                raise ValueError("Exponent too large")

        result = _BIN_OPS[op_type](left, right)

        # Guard against non-finite floats
        if isinstance(result, float):
            if result != result or result == float("inf") or result == float("-inf"):
                raise ArithmeticError("Non-finite result")

        return result


def evaluate_safe_expression(expr: str) -> Any:
    """
    Evaluate a validated arithmetic expression string and return the result.
    On any computation or parsing issue, returns a string error message.
    """
    if not isinstance(expr, str):
        return "Computation error"

    s = expr.strip()
    if not s:
        return "Computation error"

    # Ensure expression is valid before evaluating
    if not validate_expression(s):
        return "Computation error"

    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        return "Computation error"

    evaluator = _SafeExpressionEvaluator()
    try:
        return evaluator.visit(tree)
    except Exception:
        return "Computation error"
