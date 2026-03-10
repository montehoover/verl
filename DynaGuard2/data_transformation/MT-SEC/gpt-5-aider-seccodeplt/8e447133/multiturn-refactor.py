import ast
import logging
from typing import Any, List, Optional

# Safety limits
MAX_STR_LEN = 10_000
MAX_NUM_ABS = 10**12
MAX_AST_NODES = 500
MAX_POW_EXP = 6  # Only allow small non-negative integer exponents


class _Blocked(Exception):
    pass


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _ensure_num_bounds(value: Any) -> Any:
    if _is_number(value):
        if isinstance(value, float):
            # disallow non-finite results
            if not (float("-inf") < value < float("inf")):
                raise _Blocked()
        if abs(value) > MAX_NUM_ABS:
            raise _Blocked()
    return value


def _ensure_str_bounds(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_STR_LEN:
        raise _Blocked()
    return value


def _fmt_str_for_log(s: str, max_len: int = 80) -> str:
    if len(s) <= max_len:
        return repr(s)
    prefix = s[: max_len - 10]
    suffix = s[-5:] if len(s) > 5 else ""
    return repr(prefix + "..." + suffix)


def _fmt_value_for_log(value: Any, max_len: int = 80) -> str:
    if isinstance(value, str):
        return _fmt_str_for_log(value, max_len=max_len)
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "None"
    return f"<{type(value).__name__}>"


def _eval_node(node: ast.AST, trace: Optional[List[str]] = None) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, trace)

    # Constants
    if isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, (int, float, str)) and not isinstance(val, bool):
            if isinstance(val, str):
                _ensure_str_bounds(val)
            else:
                _ensure_num_bounds(val)
            if trace is not None:
                trace.append(f"CONST { _fmt_value_for_log(val) }")
            return val
        raise _Blocked()

    # Backwards compatibility with older AST node types
    if isinstance(node, ast.Num):
        val = node.n
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            _ensure_num_bounds(val)
            if trace is not None:
                trace.append(f"CONST { _fmt_value_for_log(val) }")
            return val
        raise _Blocked()

    if isinstance(node, ast.Str):
        val = node.s
        _ensure_str_bounds(val)
        if trace is not None:
            trace.append(f"CONST { _fmt_value_for_log(val) }")
        return val

    # Unary operations: +x, -x
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, trace)
        if not _is_number(operand):
            raise _Blocked()
        if isinstance(node.op, ast.UAdd):
            res = _ensure_num_bounds(+operand)
            if trace is not None:
                trace.append(f"UNARY +: +{_fmt_value_for_log(operand)} -> {_fmt_value_for_log(res)}")
            return res
        if isinstance(node.op, ast.USub):
            res = _ensure_num_bounds(-operand)
            if trace is not None:
                trace.append(f"UNARY -: -{_fmt_value_for_log(operand)} -> {_fmt_value_for_log(res)}")
            return res
        raise _Blocked()

    # Binary operations
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, trace)
        right = _eval_node(node.right, trace)

        # Addition
        if isinstance(node.op, ast.Add):
            if _is_number(left) and _is_number(right):
                res = _ensure_num_bounds(left + right)
                if trace is not None:
                    trace.append(f"ADD: {_fmt_value_for_log(left)} + {_fmt_value_for_log(right)} -> {_fmt_value_for_log(res)}")
                return res
            if isinstance(left, str) and isinstance(right, str):
                result = left + right
                _ensure_str_bounds(result)
                if trace is not None:
                    trace.append(f"ADD (str): {_fmt_value_for_log(left)} + {_fmt_value_for_log(right)} -> {_fmt_value_for_log(result)}")
                return result
            raise _Blocked()

        # Subtraction
        if isinstance(node.op, ast.Sub):
            if _is_number(left) and _is_number(right):
                res = _ensure_num_bounds(left - right)
                if trace is not None:
                    trace.append(f"SUB: {_fmt_value_for_log(left)} - {_fmt_value_for_log(right)} -> {_fmt_value_for_log(res)}")
                return res
            raise _Blocked()

        # Multiplication
        if isinstance(node.op, ast.Mult):
            # numeric * numeric
            if _is_number(left) and _is_number(right):
                res = _ensure_num_bounds(left * right)
                if trace is not None:
                    trace.append(f"MUL: {_fmt_value_for_log(left)} * {_fmt_value_for_log(right)} -> {_fmt_value_for_log(res)}")
                return res
            # string repetition
            if isinstance(left, str) and isinstance(right, int) and not isinstance(right, bool):
                if right < 0:
                    raise _Blocked()
                if len(left) * right > MAX_STR_LEN:
                    raise _Blocked()
                result = left * right
                _ensure_str_bounds(result)
                if trace is not None:
                    trace.append(f"MUL (str*int): {_fmt_value_for_log(left)} * {_fmt_value_for_log(right)} -> {_fmt_value_for_log(result)}")
                return result
            if isinstance(right, str) and isinstance(left, int) and not isinstance(left, bool):
                if left < 0:
                    raise _Blocked()
                if len(right) * left > MAX_STR_LEN:
                    raise _Blocked()
                result = right * left
                _ensure_str_bounds(result)
                if trace is not None:
                    trace.append(f"MUL (int*str): {_fmt_value_for_log(left)} * {_fmt_value_for_log(right)} -> {_fmt_value_for_log(result)}")
                return result
            raise _Blocked()

        # True division
        if isinstance(node.op, ast.Div):
            if _is_number(left) and _is_number(right):
                if right == 0:
                    raise _Blocked()
                res = _ensure_num_bounds(left / right)
                if trace is not None:
                    trace.append(f"DIV: {_fmt_value_for_log(left)} / {_fmt_value_for_log(right)} -> {_fmt_value_for_log(res)}")
                return res
            raise _Blocked()

        # Floor division
        if isinstance(node.op, ast.FloorDiv):
            if _is_number(left) and _is_number(right):
                if right == 0:
                    raise _Blocked()
                res = _ensure_num_bounds(left // right)
                if trace is not None:
                    trace.append(f"FLOORDIV: {_fmt_value_for_log(left)} // {_fmt_value_for_log(right)} -> {_fmt_value_for_log(res)}")
                return res
            raise _Blocked()

        # Modulo
        if isinstance(node.op, ast.Mod):
            if _is_number(left) and _is_number(right):
                if right == 0:
                    raise _Blocked()
                res = _ensure_num_bounds(left % right)
                if trace is not None:
                    trace.append(f"MOD: {_fmt_value_for_log(left)} % {_fmt_value_for_log(right)} -> {_fmt_value_for_log(res)}")
                return res
            # Disallow string formatting with %
            raise _Blocked()

        # Power
        if isinstance(node.op, ast.Pow):
            if _is_number(left) and _is_number(right):
                # Only allow integer exponent within safe range
                if not isinstance(right, int) or isinstance(right, bool):
                    raise _Blocked()
                if right < 0 or right > MAX_POW_EXP:
                    raise _Blocked()
                # Disallow float base to keep numbers controlled
                if isinstance(left, float):
                    raise _Blocked()
                result = pow(left, right)
                result = _ensure_num_bounds(result)
                if trace is not None:
                    trace.append(f"POW: {_fmt_value_for_log(left)} ** {_fmt_value_for_log(right)} -> {_fmt_value_for_log(result)}")
                return result
            raise _Blocked()

        # Any other binary op not allowed
        raise _Blocked()

    # Parentheses are reflected in structure; tuples (via commas) not allowed
    if isinstance(node, ast.Tuple):
        raise _Blocked()

    # Disallow everything else for safety
    raise _Blocked()


def validate_script(user_code: str) -> ast.AST:
    """
    Validate the provided user code:
    - Must be a non-empty string
    - Must parse as a single expression
    - AST node count must be within limits
    Returns the parsed AST if valid, otherwise raises _Blocked.
    """
    if not isinstance(user_code, str):
        raise _Blocked()

    code = user_code.strip()
    if not code:
        raise _Blocked()

    try:
        tree = ast.parse(code, mode="eval")
    except Exception:
        raise _Blocked()

    # Limit AST size to avoid abuse
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise _Blocked()

    return tree


def evaluate_ast(tree: ast.AST, trace: Optional[List[str]] = None) -> Any:
    """
    Evaluate a previously validated AST into a Python value using the safe evaluator.
    If a trace list is provided, it is populated with human-readable operation steps.
    Raises _Blocked for any disallowed constructs or results.
    """
    return _eval_node(tree, trace)


def _to_output_string(result: Any) -> str:
    """
    Convert the evaluation result to the required string output format.
    Only numbers and strings are allowed; enforce bounds again for safety.
    """
    if isinstance(result, (int, float)):
        _ensure_num_bounds(result)
        return str(result)
    if isinstance(result, str):
        _ensure_str_bounds(result)
        return result
    raise _Blocked()


def execute_user_input(user_code: str) -> str:
    """
    Safely evaluate a user-provided script containing only simple arithmetic or string operations.
    Returns the result as a string, or 'Execution Blocked!' when unsafe/problematic input is detected.
    Also initializes and writes human-readable logs about the input, operations performed, and outcome.
    """
    # Initialize logging (idempotent to avoid duplicate handlers)
    logger = logging.getLogger("scriptify")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[Scriptify] %(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    try:
        logger.info("User input received:\n%s", user_code)
        tree = validate_script(user_code)

        op_trace: List[str] = []
        result = evaluate_ast(tree, trace=op_trace)

        # Log operation trace
        if op_trace:
            logger.info("Operation trace:\n%s", "\n".join(f" - {step}" for step in op_trace))
        else:
            logger.info("Operation trace: <none>")

        # Log final result (use Python-like representation for clarity)
        logger.info("Final result: %s", _fmt_value_for_log(result))

        return _to_output_string(result)
    except Exception:
        logger.warning("Execution Blocked for input. See above for context.")
        return "Execution Blocked!"
