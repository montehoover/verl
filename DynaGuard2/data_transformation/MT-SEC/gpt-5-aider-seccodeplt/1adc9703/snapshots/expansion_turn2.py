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


class _SafeExprEvaluator(ast.NodeVisitor):
    """
    Evaluate a validated arithmetic AST safely without using eval().
    Supports:
      - int and float (finite)
      - +, -, *, /, %, //, unary +/-
    Enforces:
      - max node count (via external validator)
      - max integer bit length to avoid resource blowups
      - finite float results
    """

    def __init__(self, max_int_bits: int = 10_000) -> None:
        self._max_int_bits = max_int_bits

    def _ensure_numeric(self, value: Any) -> None:
        if not isinstance(value, (int, float)):
            raise ValueError("Non-numeric value encountered")

    def _check_number(self, value: Any) -> Any:
        if isinstance(value, int):
            if value.bit_length() > self._max_int_bits:
                raise ValueError("Integer result too large")
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("Non-finite float result")
            return value
        raise ValueError("Non-numeric result")

    # Root
    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    # Constants
    def visit_Constant(self, node: ast.Constant) -> Any:
        value = node.value
        if type(value) is int:
            return self._check_number(value)
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("Non-finite float not allowed")
            return value
        raise ValueError("Only int and float literals are allowed")

    def visit_Num(self, node: ast.Num) -> Any:  # type: ignore[attr-defined]
        value = node.n  # type: ignore[attr-defined]
        if type(value) is int:
            return self._check_number(value)
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("Non-finite float not allowed")
            return value
        raise ValueError("Only int and float literals are allowed")

    # Unary ops
    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        self._ensure_numeric(operand)
        if isinstance(node.op, ast.UAdd):
            return self._check_number(+operand)
        if isinstance(node.op, ast.USub):
            return self._check_number(-operand)
        raise ValueError("Only unary plus/minus are allowed")

    # Binary ops
    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        self._ensure_numeric(left)
        self._ensure_numeric(right)

        if isinstance(node.op, ast.Add):
            return self._check_number(left + right)
        if isinstance(node.op, ast.Sub):
            return self._check_number(left - right)
        if isinstance(node.op, ast.Mult):
            return self._check_number(left * right)
        if isinstance(node.op, ast.Div):
            # Python will raise ZeroDivisionError for division by zero
            return self._check_number(left / right)
        if isinstance(node.op, ast.Mod):
            return self._check_number(left % right)
        if isinstance(node.op, ast.FloorDiv):
            return self._check_number(left // right)

        raise ValueError("Only basic arithmetic operators are allowed")

    # Anything else is disallowed
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
        raise ValueError(f"Disallowed syntax: {type(node).__name__}")


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


def evaluate_safe_expression(expression: str):
    """
    Evaluate a validated arithmetic expression safely.

    Returns:
        - The numeric result (int or float) on success.
        - A string error message indicating a potential safety violation on failure.
    """
    if not isinstance(expression, str) or expression.strip() == "":
        return "Potential safety violation: invalid input"

    # Parse
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return "Potential safety violation: invalid or unsafe expression"

    # Validate
    try:
        validator = _SafeExprValidator(max_nodes=1000)
        validator.visit(tree)
    except Exception as e:
        return f"Potential safety violation: {e}"

    # Evaluate
    try:
        evaluator = _SafeExprEvaluator(max_int_bits=10_000)
        result = evaluator.visit(tree.body)
        # Final sanity check
        if isinstance(result, float) and not math.isfinite(result):
            return "Potential safety violation: non-finite result"
        if isinstance(result, int) and result.bit_length() > 10_000:
            return "Potential safety violation: result too large"
        return result
    except Exception as e:
        return f"Potential safety violation: {e}"
