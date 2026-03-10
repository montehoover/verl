import ast
from typing import Union


_Number = Union[int, float]


class _SafeCalcEvaluator(ast.NodeVisitor):
    """
    Evaluate a restricted subset of Python arithmetic expressions safely.
    Allowed:
      - numeric literals (int, float)
      - binary ops: +, -, *, /, //, %, **
      - unary ops: +, -
      - parentheses (implicitly via AST structure)
    Everything else is rejected.
    """
    __slots__ = ("_node_count", "_max_nodes", "_max_pow_exp")

    def __init__(self) -> None:
        self._node_count = 0
        self._max_nodes = 1000
        self._max_pow_exp = 100  # limit exponent magnitude to avoid resource abuse

    def _bump(self) -> None:
        self._node_count += 1
        if self._node_count > self._max_nodes:
            raise ValueError("Expression too complex")

    def visit(self, node):  # type: ignore[override]
        self._bump()
        return super().visit(node)

    def evaluate(self, node: ast.AST) -> _Number:
        # Entry point
        if isinstance(node, ast.Expression):
            return self.evaluate(node.body)
        return self.visit(node)

    def visit_Constant(self, node: ast.Constant) -> _Number:  # Python 3.8+
        val = node.value
        if isinstance(val, bool):
            # Disallow True/False (bool is subclass of int)
            raise ValueError("Booleans not allowed")
        if isinstance(val, (int, float)):
            return val
        raise ValueError("Only numeric literals allowed")

    def visit_Num(self, node: ast.Num) -> _Number:  # for older AST compatibility
        val = node.n
        if isinstance(val, bool):
            raise ValueError("Booleans not allowed")
        if isinstance(val, (int, float)):
            return val
        raise ValueError("Only numeric literals allowed")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> _Number:
        operand = self.evaluate(node.operand)
        if not isinstance(operand, (int, float)) or isinstance(operand, bool):
            raise ValueError("Invalid operand")
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError("Unsupported unary operator")

    def visit_BinOp(self, node: ast.BinOp) -> _Number:
        left = self.evaluate(node.left)
        right = self.evaluate(node.right)

        if (not isinstance(left, (int, float)) or isinstance(left, bool) or
                not isinstance(right, (int, float)) or isinstance(right, bool)):
            raise ValueError("Invalid operands")

        op = node.op
        try:
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.Div):
                return left / right
            if isinstance(op, ast.FloorDiv):
                return left // right
            if isinstance(op, ast.Mod):
                return left % right
            if isinstance(op, ast.Pow):
                # Limit exponent magnitude to prevent excessive computation/memory
                if isinstance(right, (int, float)) and abs(right) > self._max_pow_exp:
                    raise ValueError("Exponent too large")
                return left ** right
        except Exception as exc:
            # Any runtime arithmetic error results in failure
            raise ValueError(str(exc)) from exc

        raise ValueError("Unsupported binary operator")

    # Explicitly forbid all other nodes
    def generic_visit(self, node):
        raise ValueError("Unsupported syntax")


def exec_calculation(calc_string: str) -> str:
    """
    Evaluate a user-provided arithmetic expression safely.

    Args:
        calc_string: str containing only basic arithmetic (numbers and + - * / // % ** and parentheses)

    Returns:
        str: result of the evaluated expression, or 'Computation Error!' on failure or harmful input.
    """
    try:
        if not isinstance(calc_string, str):
            return 'Computation Error!'
        # Strip leading/trailing whitespace to avoid parse issues
        code = calc_string.strip()
        if not code:
            return 'Computation Error!'

        # Parse expression in 'eval' mode
        tree = ast.parse(code, mode='eval')

        evaluator = _SafeCalcEvaluator()
        result = evaluator.evaluate(tree)

        # Convert to string for output
        return str(result)
    except Exception:
        return 'Computation Error!'
