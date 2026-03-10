import ast
from typing import Union


_Number = Union[int, float]


class _ValidationVisitor(ast.NodeVisitor):
    """
    Validate that an AST contains only allowed arithmetic constructs.
    Allowed:
      - numeric literals (int, float)
      - binary ops: +, -, *, /, //, %, **
      - unary ops: +, -
      - parentheses (via AST structure)
    Everything else is rejected.
    """
    __slots__ = ("_node_count", "_max_nodes", "_max_pow_exp")

    def __init__(self) -> None:
        self._node_count = 0
        self._max_nodes = 1000
        self._max_pow_exp = 100

    def _bump(self) -> None:
        self._node_count += 1
        if self._node_count > self._max_nodes:
            raise ValueError("Expression too complex")

    def visit(self, node):  # type: ignore[override]
        self._bump()
        return super().visit(node)

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant):
        val = node.value
        if isinstance(val, bool):
            raise ValueError("Booleans not allowed")
        if not isinstance(val, (int, float)):
            raise ValueError("Only numeric literals allowed")

    def visit_Num(self, node: ast.Num):
        val = node.n
        if isinstance(val, bool):
            raise ValueError("Booleans not allowed")
        if not isinstance(val, (int, float)):
            raise ValueError("Only numeric literals allowed")

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise ValueError("Unsupported unary operator")
        self.visit(node.operand)

    def visit_BinOp(self, node: ast.BinOp):
        if not isinstance(
            node.op,
            (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow),
        ):
            raise ValueError("Unsupported binary operator")
        # Optional static sanity check for exponent if it's a literal
        if isinstance(node.op, ast.Pow):
            right = node.right
            if isinstance(right, ast.Constant) and isinstance(right.value, (int, float)):
                if abs(right.value) > self._max_pow_exp:
                    raise ValueError("Exponent too large")
            elif isinstance(right, ast.Num) and isinstance(right.n, (int, float)):
                if abs(right.n) > self._max_pow_exp:
                    raise ValueError("Exponent too large")
        self.visit(node.left)
        self.visit(node.right)

    def generic_visit(self, node):
        raise ValueError("Unsupported syntax")


class _EvalVisitor(ast.NodeVisitor):
    """
    Evaluate a validated arithmetic AST.
    Assumes the tree has been validated to only contain allowed nodes.
    """
    __slots__ = ("_node_count", "_max_nodes", "_max_pow_exp")

    def __init__(self) -> None:
        self._node_count = 0
        self._max_nodes = 1000
        self._max_pow_exp = 100

    def _bump(self) -> None:
        self._node_count += 1
        if self._node_count > self._max_nodes:
            raise ValueError("Expression too complex")

    def visit(self, node):  # type: ignore[override]
        self._bump()
        return super().visit(node)

    def evaluate(self, node: ast.AST) -> _Number:
        if isinstance(node, ast.Expression):
            return self.evaluate(node.body)
        return self.visit(node)

    def visit_Expression(self, node: ast.Expression) -> _Number:
        return self.evaluate(node.body)

    def visit_Constant(self, node: ast.Constant) -> _Number:
        val = node.value
        if isinstance(val, bool):
            raise ValueError("Booleans not allowed")
        if isinstance(val, (int, float)):
            return val
        raise ValueError("Only numeric literals allowed")

    def visit_Num(self, node: ast.Num) -> _Number:
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
                if abs(right) > self._max_pow_exp:
                    raise ValueError("Exponent too large")
                return left ** right
        except Exception as exc:
            raise ValueError(str(exc)) from exc

        raise ValueError("Unsupported binary operator")

    def generic_visit(self, node):
        raise ValueError("Unsupported syntax")


def parse_calc_string(calc_string: str) -> ast.Expression:
    """
    Parse a calculation string to an AST Expression.
    """
    return ast.parse(calc_string, mode='eval')


def validate_calc_ast(tree: ast.AST) -> None:
    """
    Validate the AST to ensure it contains only safe arithmetic constructs.
    Raises ValueError on invalid input.
    """
    _ValidationVisitor().visit(tree)


def evaluate_calc_ast(tree: ast.AST) -> _Number:
    """
    Evaluate a previously validated AST safely and return the numeric result.
    Raises ValueError on computation errors.
    """
    return _EvalVisitor().evaluate(tree)


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
        code = calc_string.strip()
        if not code:
            return 'Computation Error!'

        tree = parse_calc_string(code)
        validate_calc_ast(tree)
        result = evaluate_calc_ast(tree)

        return str(result)
    except Exception:
        return 'Computation Error!'
