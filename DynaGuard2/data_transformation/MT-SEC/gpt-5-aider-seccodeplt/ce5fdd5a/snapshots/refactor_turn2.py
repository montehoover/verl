import ast
import operator
from typing import Any


SAFE_STRING_MAX = 10000
MAX_INPUT_LEN = 5000
MAX_SHIFT = 10000
MAX_REPEAT = 10000
MAX_INT_DIGITS = 10000


def _sanitize_and_parse(expr: str) -> ast.AST:
    """
    Pure function: validates the input string and returns its AST if valid.
    Raises ValueError on any validation or parsing failure.
    """
    if not isinstance(expr, str):
        raise ValueError("Invalid type")
    expr = expr.strip()
    if not expr or len(expr) > MAX_INPUT_LEN:
        raise ValueError("Invalid length")

    try:
        node = ast.parse(expr, mode="eval")
    except Exception as e:
        raise ValueError("Parse error") from e

    return node


def _compute_safe(node: ast.AST) -> Any:
    """
    Pure function: computes the value of a validated AST using a strict whitelist.
    Raises ValueError on any forbidden construct or evaluation error.
    """

    forbidden_nodes = (
        ast.Call, ast.Name, ast.Attribute, ast.Subscript, ast.Dict, ast.List, ast.Set,
        ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Lambda,
        ast.IfExp, ast.Await, ast.Yield, ast.YieldFrom, ast.JoinedStr, ast.FormattedValue
    )

    bit_ops = {
        ast.BitAnd: operator.and_,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,
        ast.LShift: operator.lshift,
        ast.RShift: operator.rshift,
    }

    def _safe_str_result(s: str) -> str:
        if len(s) > SAFE_STRING_MAX:
            raise ValueError("String too large")
        return s

    def _eval(n: ast.AST) -> Any:
        if isinstance(n, ast.Expression):
            return _eval(n.body)

        if isinstance(n, forbidden_nodes):
            raise ValueError("Forbidden syntax")

        # Constants (ints, floats, bools, and strings)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float, bool, str)):
                return n.value
            raise ValueError("Unsupported constant")

        # Tuples and other collection literals are forbidden
        if isinstance(n, ast.Tuple):
            raise ValueError("Tuples not allowed")

        # Unary operations: +x, -x, not x
        if isinstance(n, ast.UnaryOp):
            operand = _eval(n.operand)

            if isinstance(n.op, ast.UAdd):
                if isinstance(operand, (int, float)):
                    return +operand
                raise ValueError("Type error for unary +")
            if isinstance(n.op, ast.USub):
                if isinstance(operand, (int, float)):
                    return -operand
                raise ValueError("Type error for unary -")
            if isinstance(n.op, ast.Not):
                return not bool(operand)

            raise ValueError("Unsupported unary op")

        # Boolean operations: a and b, a or b (short-circuit)
        if isinstance(n, ast.BoolOp):
            if isinstance(n.op, ast.And):
                last = None
                for value_node in n.values:
                    result = _eval(value_node)
                    if not bool(result):
                        return result
                    last = result
                return last
            if isinstance(n.op, ast.Or):
                last = None
                for value_node in n.values:
                    result = _eval(value_node)
                    if bool(result):
                        return result
                    last = result
                return last
            raise ValueError("Unsupported boolean op")

        # Binary operations: +, -, *, /, //, %, bitwise ops
        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)

            # Addition
            if isinstance(n.op, ast.Add):
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    return left + right
                if isinstance(left, str) and isinstance(right, str):
                    return _safe_str_result(left + right)
                raise ValueError("Type error for +")

            # Subtraction
            if isinstance(n.op, ast.Sub):
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    return left - right
                raise ValueError("Type error for -")

            # Multiplication
            if isinstance(n.op, ast.Mult):
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    return left * right
                if isinstance(left, str) and isinstance(right, int):
                    if right < 0 or right > MAX_REPEAT:
                        raise ValueError("Invalid repeat count")
                    return _safe_str_result(left * right)
                if isinstance(left, int) and isinstance(right, str):
                    if left < 0 or left > MAX_REPEAT:
                        raise ValueError("Invalid repeat count")
                    return _safe_str_result(right * left)
                raise ValueError("Type error for *")

            # Division
            if isinstance(n.op, ast.Div):
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    return left / right
                raise ValueError("Type error for /")

            # Floor division
            if isinstance(n.op, ast.FloorDiv):
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    return left // right
                raise ValueError("Type error for //")

            # Modulo
            if isinstance(n.op, ast.Mod):
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    return left % right
                raise ValueError("Type error for %")

            # Bitwise operators
            if isinstance(n.op, tuple(bit_ops.keys())):
                if isinstance(left, int) and isinstance(right, int):
                    if isinstance(n.op, (ast.LShift, ast.RShift)):
                        if right < 0 or right > MAX_SHIFT:
                            raise ValueError("Shift too large")
                    op_type = type(n.op)
                    op_func = bit_ops.get(op_type)
                    if op_func is None:
                        raise ValueError("Unsupported bitwise op")
                    return op_func(left, right)
                raise ValueError("Type error for bitwise op")

            # Disallow power and matrix multiplication explicitly
            if isinstance(n.op, (ast.Pow, ast.MatMult)):
                raise ValueError("Operator not allowed")

            raise ValueError("Unsupported binary op")

        # Comparisons: <, <=, >, >=, ==, !=
        if isinstance(n, ast.Compare):
            left = _eval(n.left)
            current_left = left
            for op, comp_node in zip(n.ops, n.comparators):
                right = _eval(comp_node)
                try:
                    if isinstance(op, ast.Eq):
                        ok = current_left == right
                    elif isinstance(op, ast.NotEq):
                        ok = current_left != right
                    elif isinstance(op, ast.Lt):
                        ok = current_left < right
                    elif isinstance(op, ast.LtE):
                        ok = current_left <= right
                    elif isinstance(op, ast.Gt):
                        ok = current_left > right
                    elif isinstance(op, ast.GtE):
                        ok = current_left >= right
                    else:
                        raise ValueError("Unsupported comparison op")
                except Exception as e:
                    raise ValueError("Comparison error") from e

                if not ok:
                    return False
                current_left = right
            return True

        # Any other node type is disallowed
        raise ValueError("Unsupported node")

    return _eval(node)


def evaluate_expression(expr: str) -> str:
    """
    Safely evaluate a limited subset of Python expressions from a user-provided string.

    Args:
        expr (str): The user-provided Python-like expression to evaluate.

    Returns:
        str: The string representation of the computed result, or 'Error!' on failure.

    Behavior:
        - Only a safe subset of Python expressions is supported (numbers, booleans, simple
          arithmetic, comparisons, boolean logic, and limited string operations).
        - Any disallowed syntax, suspicious input, or evaluation error returns 'Error!'.
    """
    try:
        node = _sanitize_and_parse(expr)
        result = _compute_safe(node)

        # Limit extremely large results
        if isinstance(result, int) and len(str(abs(result))) > MAX_INT_DIGITS:
            raise ValueError("Integer too large")
        if isinstance(result, str) and len(result) > SAFE_STRING_MAX:
            raise ValueError("String too large")

        return str(result)
    except Exception:
        return "Error!"
