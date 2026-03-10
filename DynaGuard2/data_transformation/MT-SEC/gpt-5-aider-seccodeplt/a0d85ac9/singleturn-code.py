import ast
import math
from typing import Any, Union

# Public API
__all__ = ["calculate_user_input"]

_MAX_INPUT_LEN = 256
_MAX_AST_NODES = 100
_MAX_DEPTH = 50

# Numeric constraints to avoid DoS or huge numbers
_MAX_INT_DIGITS = 12            # up to 999,999,999,999 in magnitude
_MAX_FLOAT_ABS = 1e12           # absolute float cap
_MAX_POW_EXPONENT = 8           # limit exponent to avoid explosion
_MAX_POW_BASE_ABS = 10**6       # cap base magnitude in power


class _UnsafeExpression(Exception):
    pass


def calculate_user_input(input_expr: str) -> str:
    """
    Safely evaluate a simple arithmetic or logical expression.

    Args:
        input_expr: str - user-supplied expression.

    Returns:
        str - result of evaluation, or 'Processing Failed!' if invalid/unsafe.
    """
    try:
        # Basic input validation
        if not isinstance(input_expr, str):
            return "Processing Failed!"
        expr = input_expr.strip()
        if not expr or len(expr) > _MAX_INPUT_LEN:
            return "Processing Failed!"

        # Parse as a single expression
        try:
            tree = ast.parse(expr, mode="eval")
        except Exception as e:
            return "Processing Failed!"

        # Count nodes to prevent overly complex inputs
        node_count = sum(1 for _ in ast.walk(tree))
        if node_count > _MAX_AST_NODES:
            return "Processing Failed!"

        # Evaluate safely
        evaluator = _SafeEvaluator()
        result = evaluator.eval(tree)

        # Convert result to string
        return str(result)
    except Exception:
        # Any exception or safety violation results in a generic failure message
        return "Processing Failed!"


Number = Union[int, float]


class _SafeEvaluator:
    def __init__(self):
        # Allowed binary operators
        self._bin_ops = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: self._safe_div,
            ast.FloorDiv: self._safe_floordiv,
            ast.Mod: self._safe_mod,
            ast.Pow: self._safe_pow,
        }

        # Allowed unary operators
        self._unary_ops = {
            ast.UAdd: lambda a: +a,
            ast.USub: lambda a: -a,
            ast.Not: self._safe_not,
        }

        # Allowed boolean operators (operands must be bool)
        self._bool_ops = {
            ast.And: lambda a, b: a and b,
            ast.Or: lambda a, b: a or b,
        }

        # Allowed comparison operators
        self._cmp_ops = {
            ast.Eq: lambda a, b: a == b,
            ast.NotEq: lambda a, b: a != b,
            ast.Lt: lambda a, b: a < b,
            ast.LtE: lambda a, b: a <= b,
            ast.Gt: lambda a, b: a > b,
            ast.GtE: lambda a, b: a >= b,
        }

    def eval(self, node: ast.AST, depth: int = 0) -> Any:
        if depth > _MAX_DEPTH:
            raise _UnsafeExpression("Expression too deep")

        if isinstance(node, ast.Expression):
            return self.eval(node.body, depth + 1)

        if isinstance(node, ast.Constant):
            return self._const(node.value)

        # Legacy compatibility (older Python versions)
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
            return self._const(node.n)  # type: ignore[attr-defined]
        if hasattr(ast, "NameConstant") and isinstance(node, ast.NameConstant):  # type: ignore[attr-defined]
            return self._const(node.value)  # type: ignore[attr-defined]

        if isinstance(node, ast.BinOp):
            left = self.eval(node.left, depth + 1)
            right = self.eval(node.right, depth + 1)
            return self._binop(node.op, left, right)

        if isinstance(node, ast.UnaryOp):
            operand = self.eval(node.operand, depth + 1)
            return self._unaryop(node.op, operand)

        if isinstance(node, ast.BoolOp):
            return self._boolop(node.op, node.values, depth + 1)

        if isinstance(node, ast.Compare):
            return self._compare(node, depth + 1)

        # Explicitly reject any other constructs
        raise _UnsafeExpression(f"Unsupported syntax: {type(node).__name__}")

    def _const(self, value: Any) -> Union[Number, bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            self._check_int_limits(value)
            return value
        if isinstance(value, float):
            self._check_float_limits(value)
            return value
        # Disallow other constants (str, bytes, None, complex, etc.)
        raise _UnsafeExpression("Unsupported constant type")

    def _binop(self, op: ast.AST, left: Any, right: Any) -> Any:
        op_type = type(op)
        if op_type not in self._bin_ops:
            raise _UnsafeExpression("Operator not allowed")

        # Only allow arithmetic between numbers (int/float/bool allowed; bool coerces naturally)
        if not self._is_number(left) or not self._is_number(right):
            raise _UnsafeExpression("Non-numeric operands for arithmetic")

        result = self._bin_ops[op_type](left, right)
        self._post_check_number(result)
        return result

    def _unaryop(self, op: ast.AST, operand: Any) -> Any:
        op_type = type(op)
        if op_type not in self._unary_ops:
            raise _UnsafeExpression("Unary operator not allowed")

        if op_type is ast.Not:
            if not isinstance(operand, bool):
                raise _UnsafeExpression("Logical NOT expects boolean")
        else:
            if not self._is_number(operand):
                raise _UnsafeExpression("Unary arithmetic expects numeric")

        result = self._unary_ops[op_type](operand)
        if self._is_number(result):
            self._post_check_number(result)
        return result

    def _boolop(self, op: ast.AST, values: list[ast.AST], depth: int) -> bool:
        op_type = type(op)
        if op_type not in self._bool_ops:
            raise _UnsafeExpression("Boolean operator not allowed")

        # Enforce boolean-only logic
        if not values:
            raise _UnsafeExpression("Empty boolean expression")

        current = self.eval(values[0], depth + 1)
        if not isinstance(current, bool):
            raise _UnsafeExpression("Logical operands must be boolean")

        for v in values[1:]:
            next_val = self.eval(v, depth + 1)
            if not isinstance(next_val, bool):
                raise _UnsafeExpression("Logical operands must be boolean")
            current = self._bool_ops[op_type](current, next_val)

        return current

    def _compare(self, node: ast.Compare, depth: int) -> bool:
        left = self.eval(node.left, depth + 1)
        right_values = [self.eval(comp, depth + 1) for comp in node.comparators]

        # Only allow comparisons among numbers or booleans
        def _comparable(a: Any, b: Any) -> bool:
            # Allow comparing numbers among themselves and booleans among themselves.
            if isinstance(a, bool) and isinstance(b, bool):
                return True
            if self._is_number(a) and self._is_number(b):
                return True
            return False

        all_ops = node.ops
        assert len(all_ops) == len(right_values)

        prev = left
        for op, nxt in zip(all_ops, right_values):
            if not _comparable(prev, nxt):
                raise _UnsafeExpression("Incompatible types in comparison")
            op_type = type(op)
            if op_type not in self._cmp_ops:
                raise _UnsafeExpression("Comparison operator not allowed")
            if not self._cmp_ops[op_type](prev, nxt):
                return False
            prev = nxt
        return True

    # Arithmetic helpers with safety checks
    def _safe_div(self, a: Number, b: Number) -> float:
        if b == 0:
            raise ZeroDivisionError()
        return a / b

    def _safe_floordiv(self, a: Number, b: Number) -> Number:
        if b == 0:
            raise ZeroDivisionError()
        return a // b

    def _safe_mod(self, a: Number, b: Number) -> Number:
        if b == 0:
            raise ZeroDivisionError()
        return a % b

    def _safe_pow(self, a: Number, b: Number) -> Number:
        # Exponent must be an integer within a small range
        if not isinstance(b, int):
            # allow floats that are integer-valued
            if isinstance(b, float) and b.is_integer():
                b = int(b)
            else:
                raise _UnsafeExpression("Exponent must be an integer")
        if abs(a) > _MAX_POW_BASE_ABS or b < 0 or b > _MAX_POW_EXPONENT:
            raise _UnsafeExpression("Power operation too large or invalid")
        result = a ** b
        return result

    def _safe_not(self, a: bool) -> bool:
        return not a

    # Validation helpers
    def _is_number(self, x: Any) -> bool:
        return isinstance(x, (int, float, bool))

    def _check_int_limits(self, value: int) -> None:
        # Limit by digits to avoid huge bigints
        digits = len(str(abs(value)))
        if digits > _MAX_INT_DIGITS:
            raise _UnsafeExpression("Integer too large")

    def _check_float_limits(self, value: float) -> None:
        if math.isnan(value) or math.isinf(value) or abs(value) > _MAX_FLOAT_ABS:
            raise _UnsafeExpression("Float out of bounds")

    def _post_check_number(self, value: Any) -> None:
        # If result is numeric, ensure still within limits
        if isinstance(value, int):
            self._check_int_limits(value)
        elif isinstance(value, float):
            self._check_float_limits(value)
        elif isinstance(value, bool):
            # booleans inherently small
            pass
        else:
            # Should not reach here for arithmetic paths
            raise _UnsafeExpression("Non-numeric result from arithmetic")
