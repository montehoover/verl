import ast
import operator
import logging
from typing import Any, Optional, Tuple


SAFE_STRING_MAX = 10000
MAX_INPUT_LEN = 5000
MAX_SHIFT = 10000
MAX_REPEAT = 10000
MAX_INT_DIGITS = 10000

# Logger setup: logs to a file in the current directory
logger = logging.getLogger("QUIZAPP")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("quizapp_eval.log", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False


def _sanitize_and_parse(expr: Any) -> Optional[ast.AST]:
    """
    Pure function: validates the input string and returns its AST if valid.
    Returns None on any validation or parsing failure.
    """
    if not isinstance(expr, str):
        return None
    s = expr.strip()
    if not s or len(s) > MAX_INPUT_LEN:
        return None
    try:
        return ast.parse(s, mode="eval")
    except Exception:
        return None


def _compute_safe(node: ast.AST) -> Tuple[bool, Any]:
    """
    Pure function: computes the value of a validated AST using a strict whitelist.
    Returns (True, value) on success, or (False, None) on any forbidden construct or error.
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

    def _ok(value: Any) -> Tuple[bool, Any]:
        return True, value

    def _err() -> Tuple[bool, Any]:
        return False, None

    def _safe_str_result(s: str) -> Tuple[bool, Any]:
        if len(s) > SAFE_STRING_MAX:
            return _err()
        return _ok(s)

    def _eval(n: ast.AST) -> Tuple[bool, Any]:
        if isinstance(n, ast.Expression):
            return _eval(n.body)

        if isinstance(n, forbidden_nodes):
            return _err()

        # Constants (ints, floats, bools, and strings)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float, bool, str)):
                return _ok(n.value)
            return _err()

        # Tuples and other collection literals are forbidden
        if isinstance(n, ast.Tuple):
            return _err()

        # Unary operations: +x, -x, not x
        if isinstance(n, ast.UnaryOp):
            ok, operand = _eval(n.operand)
            if not ok:
                return _err()

            if isinstance(n.op, ast.UAdd):
                if isinstance(operand, (int, float)):
                    return _ok(+operand)
                return _err()
            if isinstance(n.op, ast.USub):
                if isinstance(operand, (int, float)):
                    return _ok(-operand)
                return _err()
            if isinstance(n.op, ast.Not):
                return _ok(not bool(operand))

            return _err()

        # Boolean operations: a and b, a or b (short-circuit)
        if isinstance(n, ast.BoolOp):
            if isinstance(n.op, ast.And):
                last: Any = True
                for value_node in n.values:
                    ok, result = _eval(value_node)
                    if not ok:
                        return _err()
                    if not bool(result):
                        return _ok(result)
                    last = result
                return _ok(last)
            if isinstance(n.op, ast.Or):
                last: Any = None
                for value_node in n.values:
                    ok, result = _eval(value_node)
                    if not ok:
                        return _err()
                    if bool(result):
                        return _ok(result)
                    last = result
                return _ok(last)
            return _err()

        # Binary operations: +, -, *, /, //, %, bitwise ops
        if isinstance(n, ast.BinOp):
            ok, left = _eval(n.left)
            if not ok:
                return _err()
            ok, right = _eval(n.right)
            if not ok:
                return _err()

            # Addition
            if isinstance(n.op, ast.Add):
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    return _ok(left + right)
                if isinstance(left, str) and isinstance(right, str):
                    return _safe_str_result(left + right)
                return _err()

            # Subtraction
            if isinstance(n.op, ast.Sub):
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    return _ok(left - right)
                return _err()

            # Multiplication
            if isinstance(n.op, ast.Mult):
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    return _ok(left * right)
                if isinstance(left, str) and isinstance(right, int):
                    if right < 0 or right > MAX_REPEAT:
                        return _err()
                    return _safe_str_result(left * right)
                if isinstance(left, int) and isinstance(right, str):
                    if left < 0 or left > MAX_REPEAT:
                        return _err()
                    return _safe_str_result(right * left)
                return _err()

            # Division
            if isinstance(n.op, ast.Div):
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    try:
                        return _ok(left / right)
                    except Exception:
                        return _err()
                return _err()

            # Floor division
            if isinstance(n.op, ast.FloorDiv):
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    try:
                        return _ok(left // right)
                    except Exception:
                        return _err()
                return _err()

            # Modulo
            if isinstance(n.op, ast.Mod):
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    try:
                        return _ok(left % right)
                    except Exception:
                        return _err()
                return _err()

            # Bitwise operators
            if isinstance(n.op, tuple(bit_ops.keys())):
                if isinstance(left, int) and isinstance(right, int):
                    if isinstance(n.op, (ast.LShift, ast.RShift)):
                        if right < 0 or right > MAX_SHIFT:
                            return _err()
                    op_type = type(n.op)
                    op_func = bit_ops.get(op_type)
                    if op_func is None:
                        return _err()
                    try:
                        return _ok(op_func(left, right))
                    except Exception:
                        return _err()
                return _err()

            # Disallow power and matrix multiplication explicitly
            if isinstance(n.op, (ast.Pow, ast.MatMult)):
                return _err()

            return _err()

        # Comparisons: <, <=, >, >=, ==, !=
        if isinstance(n, ast.Compare):
            ok, left = _eval(n.left)
            if not ok:
                return _err()
            current_left = left
            for op, comp_node in zip(n.ops, n.comparators):
                ok, right = _eval(comp_node)
                if not ok:
                    return _err()
                try:
                    if isinstance(op, ast.Eq):
                        ok_cmp = current_left == right
                    elif isinstance(op, ast.NotEq):
                        ok_cmp = current_left != right
                    elif isinstance(op, ast.Lt):
                        ok_cmp = current_left < right
                    elif isinstance(op, ast.LtE):
                        ok_cmp = current_left <= right
                    elif isinstance(op, ast.Gt):
                        ok_cmp = current_left > right
                    elif isinstance(op, ast.GtE):
                        ok_cmp = current_left >= right
                    else:
                        return _err()
                except Exception:
                    return _err()

                if not ok_cmp:
                    return _ok(False)
                current_left = right
            return _ok(True)

        # Any other node type is disallowed
        return _err()

    try:
        return _eval(node)
    except Exception:
        return False, None


def evaluate_expression(expr: str) -> str:
    """
    Safely evaluate a limited subset of Python expressions from a user-provided string.

    Args:
        expr (str): The user-provided Python-like expression to evaluate.

    Returns:
        str: The string representation of the computed result, or 'Error!' on failure.
    """
    try:
        node = _sanitize_and_parse(expr)
        if node is None:
            logger.info("expr=%r result=%s", expr, "Error!")
            return "Error!"

        ok, result = _compute_safe(node)
        if not ok:
            logger.info("expr=%r result=%s", expr, "Error!")
            return "Error!"

        # Limit extremely large results
        if isinstance(result, int) and len(str(abs(result))) > MAX_INT_DIGITS:
            logger.info("expr=%r result=%s", expr, "Error!")
            return "Error!"
        if isinstance(result, str) and len(result) > SAFE_STRING_MAX:
            logger.info("expr=%r result=%s", expr, "Error!")
            return "Error!"

        out = str(result)
        logger.info("expr=%r result=%s", expr, out)
        return out
    except Exception:
        # Any unexpected exception results in an error and is logged.
        try:
            logger.info("expr=%r result=%s", expr, "Error!")
        except Exception:
            pass
        return "Error!"
