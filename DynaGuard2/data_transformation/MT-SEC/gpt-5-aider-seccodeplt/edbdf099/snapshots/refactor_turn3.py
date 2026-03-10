import ast
import math
import logging
import time
from typing import Any, Union


__all__ = ["process_user_code"]


# Safety limits
_MAX_SCRIPT_LENGTH = 5000
_MAX_AST_NODES = 100
_MAX_STRING_CONST_LENGTH = 10000
_MAX_RESULT_LENGTH = 20000
_MAX_STRING_REPEAT = 10000
_MAX_INT_CONST_ABS = 10 ** 12
_MAX_INT_RESULT_ABS = 10 ** 18


# Logger setup
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _SafeEvalError(Exception):
    pass


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float))


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        # Python ints are unbounded; we'll limit magnitude.
        return abs(value) <= _MAX_INT_RESULT_ABS
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _validate_constant(value: Any) -> Any:
    # Allow only str, int, float; with size checks
    if isinstance(value, bool):
        # bools are not considered basic arithmetic here
        raise _SafeEvalError("Booleans not allowed")
    if isinstance(value, int):
        if abs(value) > _MAX_INT_CONST_ABS:
            raise _SafeEvalError("Integer constant too large")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _SafeEvalError("Non-finite float")
        return value
    if isinstance(value, str):
        if len(value) > _MAX_STRING_CONST_LENGTH:
            raise _SafeEvalError("String constant too long")
        return value
    raise _SafeEvalError("Unsupported constant type")


def _safe_binop_add(left: Any, right: Any) -> Any:
    # number + number
    if _is_number(left) and _is_number(right):
        return left + right
    # string + string
    if isinstance(left, str) and isinstance(right, str):
        combined_len = len(left) + len(right)
        if combined_len > _MAX_RESULT_LENGTH:
            raise _SafeEvalError("Resulting string too long")
        return left + right
    raise _SafeEvalError("Unsupported operands for +")


def _safe_binop_sub(left: Any, right: Any) -> Any:
    if _is_number(left) and _is_number(right):
        return left - right
    raise _SafeEvalError("Unsupported operands for -")


def _safe_binop_mul(left: Any, right: Any) -> Any:
    # number * number
    if _is_number(left) and _is_number(right):
        return left * right
    # string * int
    if isinstance(left, str) and isinstance(right, int):
        if right < 0 or right > _MAX_STRING_REPEAT:
            raise _SafeEvalError("Invalid repetition count")
        if len(left) * right > _MAX_RESULT_LENGTH:
            raise _SafeEvalError("Resulting string too long")
        return left * right
    # int * string
    if isinstance(left, int) and isinstance(right, str):
        if left < 0 or left > _MAX_STRING_REPEAT:
            raise _SafeEvalError("Invalid repetition count")
        if len(right) * left > _MAX_RESULT_LENGTH:
            raise _SafeEvalError("Resulting string too long")
        return right * left
    raise _SafeEvalError("Unsupported operands for *")


def _safe_binop_div(left: Any, right: Any) -> Any:
    if _is_number(left) and _is_number(right):
        if right == 0:
            raise _SafeEvalError("Division by zero")
        return left / right
    raise _SafeEvalError("Unsupported operands for /")


def _safe_binop_floordiv(left: Any, right: Any) -> Any:
    if _is_number(left) and _is_number(right):
        if right == 0:
            raise _SafeEvalError("Division by zero")
        return left // right
    raise _SafeEvalError("Unsupported operands for //")


def _safe_binop_mod(left: Any, right: Any) -> Any:
    if _is_number(left) and _is_number(right):
        if right == 0:
            raise _SafeEvalError("Modulo by zero")
        return left % right
    # Explicitly disallow string formatting with %
    raise _SafeEvalError("Unsupported operands for %")


_ALLOWED_BINOPS = {
    ast.Add: _safe_binop_add,
    ast.Sub: _safe_binop_sub,
    ast.Mult: _safe_binop_mul,
    ast.Div: _safe_binop_div,
    ast.FloorDiv: _safe_binop_floordiv,
    ast.Mod: _safe_binop_mod,
    # Intentionally disallow Pow, BitAnd/Or/Xor, Shifts, MatMult, etc.
}


def _eval_binop(node: ast.BinOp) -> Any:
    op_type = type(node.op)
    if op_type not in _ALLOWED_BINOPS:
        raise _SafeEvalError("Operator not allowed")
    left = _safe_eval(node.left)
    right = _safe_eval(node.right)
    result = _ALLOWED_BINOPS[op_type](left, right)
    return result


def _eval_unaryop(node: ast.UnaryOp) -> Any:
    operand = _safe_eval(node.operand)
    if isinstance(node.op, ast.UAdd):
        if not _is_number(operand):
            raise _SafeEvalError("Unary + allowed only on numbers")
        return +operand
    if isinstance(node.op, ast.USub):
        if not _is_number(operand):
            raise _SafeEvalError("Unary - allowed only on numbers")
        return -operand
    # Disallow bitwise invert ~
    raise _SafeEvalError("Unary operator not allowed")


def _safe_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    # Constants
    if isinstance(node, ast.Constant):
        return _validate_constant(node.value)
    # For older Python versions
    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
        return _validate_constant(node.n)  # type: ignore[attr-defined]
    if hasattr(ast, "Str") and isinstance(node, ast.Str):  # type: ignore[attr-defined]
        return _validate_constant(node.s)  # type: ignore[attr-defined]

    # Arithmetic
    if isinstance(node, ast.BinOp):
        return _eval_binop(node)
    if isinstance(node, ast.UnaryOp):
        return _eval_unaryop(node)

    # Parentheses are represented implicitly; no dedicated node to allow.

    # Explicitly block any of these by default:
    disallowed_nodes = (
        ast.Call,
        ast.Attribute,
        ast.Subscript,
        ast.Slice,
        ast.IfExp,
        ast.Dict,
        ast.List,
        ast.Set,
        ast.Tuple,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp if hasattr(ast, "DictComp") else ast.Dict,  # type: ignore
        ast.GeneratorExp,
        ast.Lambda,
        ast.Compare,
        ast.BoolOp,
        ast.Name,
        ast.FormattedValue,
        ast.JoinedStr,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
        ast.Assign,
        ast.AugAssign,
        ast.AnnAssign,
        ast.Starred,
    )
    if isinstance(node, disallowed_nodes):
        raise _SafeEvalError("Disallowed syntax")

    # Any other node is not allowed
    raise _SafeEvalError("Unsupported syntax")


# -------------------
# Validation helpers
# -------------------

def _validate_code_script(code_script: str) -> None:
    """
    Pure validation of the raw script string (type and length only).
    Raises _SafeEvalError on validation failure.
    """
    if not isinstance(code_script, str):
        raise _SafeEvalError("Invalid script type")
    if len(code_script) > _MAX_SCRIPT_LENGTH:
        raise _SafeEvalError("Script too long")


def _parse_as_expression(code_script: str) -> ast.Expression:
    """
    Parse code as a Python expression. Raises _SafeEvalError if parsing fails.
    """
    try:
        tree = ast.parse(code_script, mode="eval")
        assert isinstance(tree, ast.Expression)
        return tree
    except Exception as ex:
        raise _SafeEvalError("Parse error") from ex


def _limit_ast_size(tree: ast.AST) -> None:
    """
    Ensure the AST is not excessively large.
    """
    node_count = sum(1 for _ in ast.walk(tree))
    if node_count > _MAX_AST_NODES:
        raise _SafeEvalError("AST too large")


def _validate_and_parse_script(code_script: str) -> ast.Expression:
    """
    Pure function to validate script string, parse to AST Expression, and
    enforce AST-level limits. Raises _SafeEvalError on any failure.
    """
    _validate_code_script(code_script)
    tree = _parse_as_expression(code_script)
    _limit_ast_size(tree)
    return tree


# -------------------
# Execution helpers
# -------------------

def _execute_ast(tree: ast.AST) -> Any:
    """
    Pure evaluation of a pre-validated AST using the safe evaluator.
    """
    return _safe_eval(tree)


def _format_result(result: Any) -> str:
    """
    Validate and format the evaluation result into a string.
    Raises _SafeEvalError if the result is invalid or unsafe.
    """
    if isinstance(result, str):
        if len(result) > _MAX_RESULT_LENGTH:
            raise _SafeEvalError("Result too long")
        return result

    if _is_number(result):
        if not _is_finite_number(result):
            raise _SafeEvalError("Non-finite or too-large number")
        return str(result)

    raise _SafeEvalError("Unsupported result type")


def _execute_script(tree: ast.Expression) -> str:
    """
    Execute a validated AST and return the formatted string result.
    """
    result = _execute_ast(tree)
    return _format_result(result)


def _preview(text: str, limit: int = 200) -> str:
    """
    Create a safe preview of text for logging.
    """
    return text[:limit] + ("..." if len(text) > limit else "")


def process_user_code(code_script: str) -> str:
    """
    Securely parse and evaluate a small Python expression that performs basic
    arithmetic or string operations.

    Args:
        code_script: str - a Python expression containing simple arithmetic or string manipulations.

    Returns:
        str - evaluated result of the expression, or 'Execution Blocked!' if the script
              contains dangerous or unsupported actions.

    Notes:
        - Only expressions are allowed (no statements).
        - Allowed operations: +, -, *, /, //, % for numbers; + for strings; string * int.
        - Disallows function calls, attribute access, subscripting, comprehensions, etc.
    """
    start_ts = time.time()
    outcome = "blocked"
    reason = None
    result_str = "Execution Blocked!"
    # Attempt log
    try:
        if isinstance(code_script, str):
            logger.info(
                "process_user_code.attempt length=%s preview=%r",
                len(code_script),
                _preview(code_script),
            )
        else:
            logger.info("process_user_code.attempt length=N/A preview=<non-str>")
        tree = _validate_and_parse_script(code_script)
        result_str = _execute_script(tree)
        outcome = "success"
        return result_str
    except _SafeEvalError as e:
        reason = str(e)
        return "Execution Blocked!"
    except Exception:
        reason = "unexpected_error"
        logger.exception("process_user_code.unexpected_error")
        return "Execution Blocked!"
    finally:
        duration_ms = int((time.time() - start_ts) * 1000)
        if outcome == "success":
            logger.info(
                "process_user_code.result status=success duration_ms=%d result_len=%d result_preview=%r",
                duration_ms,
                len(result_str),
                _preview(result_str),
            )
        else:
            logger.warning(
                "process_user_code.result status=blocked duration_ms=%d reason=%s",
                duration_ms,
                reason or "blocked",
            )
