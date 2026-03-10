"""
TASKMANAGER safe expression processor.

This module exposes a single public function, process_user_expression, which
safely evaluates basic user-provided logical or mathematical expressions.
It aims to prevent code execution, attribute access, name lookup, or other
unsafe constructs by using Python's AST to strictly whitelist allowed nodes
and operators, then performing a controlled evaluation.

Only the following constructs are supported:
- Numeric literals (int, float) within configured bounds
- Boolean literals (True, False)
- Arithmetic operators: +, -, *, /, //, %, **
- Unary operators: +x, -x, not x
- Boolean operators: and, or (strictly boolean operands)
- Comparisons: ==, !=, <, <=, >, >=
- Conditional expression: a if cond else b

Any unsupported or unsafe input results in 'Processing Failed!'.

Logging:
- This module logs each expression processed, including whether it succeeded
  or failed, and the resulting value on success. Parsing errors, unsafe
  constructs, and unexpected exceptions are logged at appropriate levels.
"""

from typing import Any, Tuple, Union
import logging
import ast
import math
import operator

# Module-level logger for this component. A NullHandler is added so that
# importing applications can configure logging as they see fit without noise.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Configuration limits to avoid abuse or resource exhaustion.
_MAX_EXPR_LEN = 500
_MAX_DEPTH = 25
_MAX_ABS_INT = 10 ** 12
_MAX_ABS_FLOAT = 1e12
_MAX_POW_EXPONENT = 10
_MAX_POW_BASE_ABS = 10 ** 6


class _UnsafeExpression(Exception):
    """Internal exception to signal unsafe or invalid user expressions."""
    pass


def process_user_expression(expr: str) -> str:
    """
    Safely process a user-provided logical or mathematical expression.

    The expression is parsed into an AST, validated against a strict whitelist,
    and evaluated in a controlled manner. Any error (syntax, type, range, or
    security) results in the string 'Processing Failed!'.

    Logging:
        - Logs the start of processing and the expression string (at INFO).
        - Logs parsing errors and validation failures (at WARNING).
        - Logs unexpected exceptions with traceback (at ERROR).
        - Logs successful evaluations with the resulting value (at INFO).

    Args:
        expr: A string containing the user's expression to evaluate.

    Returns:
        str: The evaluation result converted to string, or 'Processing Failed!'
             if the input is invalid or unsafe.

    Notes:
        - Names, function calls, attribute access, subscripts, comprehensions,
          and statements are not allowed.
        - Numbers outside configured bounds are rejected.
        - Exponentiation is limited in base and exponent size.
    """
    # Log the incoming expression (length is constrained by _MAX_EXPR_LEN).
    logger.info("Processing expression: %s", expr)

    try:
        if not isinstance(expr, str):
            logger.warning(
                "Rejected expression due to invalid type: %r", type(expr).__name__
            )
            return 'Processing Failed!'

        if len(expr) == 0 or len(expr) > _MAX_EXPR_LEN:
            logger.warning(
                "Rejected expression due to invalid length: len=%d (max=%d)",
                len(expr),
                _MAX_EXPR_LEN,
            )
            return 'Processing Failed!'

        # Parse expression in evaluation mode (no statements allowed).
        try:
            tree = ast.parse(expr, mode='eval')
        except SyntaxError as exc:
            logger.warning(
                "Syntax error while parsing expression: %s; error=%s", expr, exc
            )
            return 'Processing Failed!'

        # Evaluate the root expression node with depth tracking.
        result = _eval_node(tree.body, depth=0)

        # Only primitive results are allowed.
        if not isinstance(result, (int, float, bool)):
            logger.warning(
                "Rejected non-primitive result for expression: %s; type=%s",
                expr,
                type(result).__name__,
            )
            return 'Processing Failed!'

        # Final sanity checks for numeric ranges.
        if isinstance(result, bool):
            # Booleans are allowed as-is.
            pass
        elif isinstance(result, int):
            if abs(result) > _MAX_ABS_INT:
                logger.warning(
                    "Result out of bounds (int): %s => %s (max abs=%s)",
                    expr,
                    result,
                    _MAX_ABS_INT,
                )
                return 'Processing Failed!'
        elif isinstance(result, float):
            if not math.isfinite(result) or abs(result) > _MAX_ABS_FLOAT:
                logger.warning(
                    "Result out of bounds (float/non-finite): %s => %s (max abs=%s)",
                    expr,
                    result,
                    _MAX_ABS_FLOAT,
                )
                return 'Processing Failed!'

        logger.info("Expression evaluated successfully: %s => %s", expr, result)
        return str(result)

    except _UnsafeExpression as exc:
        logger.warning(
            "Unsafe or invalid expression rejected: %s; reason=%s", expr, exc
        )
        return 'Processing Failed!'

    except Exception:
        # Any unhandled error must not leak details to the caller.
        logger.exception("Unexpected error while processing expression: %s", expr)
        return 'Processing Failed!'


# Mapping of allowed binary operators to their safe implementations.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Mapping of allowed unary numeric operators.
_UNARY_NUMERIC = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Allowed comparison operator node types.
_CMP_OPS = (
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)

# Node types that are always considered unsafe in this context.
_UNSAFE_NODE_TYPES_LIST = [
    ast.Name,
    ast.Call,
    ast.Attribute,
    ast.Subscript,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.GeneratorExp,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.If,
    ast.For,
    ast.While,
    ast.With,
    ast.Return,
    ast.Delete,
    ast.Try,
    ast.Raise,
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.Assert,
    ast.Pass,
    ast.Break,
    ast.Continue,
]
# Add DictComp dynamically for interpreter versions that define it.
if hasattr(ast, "DictComp"):
    _UNSAFE_NODE_TYPES_LIST.append(ast.DictComp)

# Convert to a tuple for isinstance checks.
_UNSAFE_NODE_TYPES: Tuple[type, ...] = tuple(_UNSAFE_NODE_TYPES_LIST)


def _is_number(x: Any) -> bool:
    """
    Determine whether a value is a number (int or float), excluding booleans.

    Args:
        x: Value to test.

    Returns:
        True if x is an int or float and not a bool, else False.
    """
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_number_bounds(x: Union[int, float]) -> None:
    """
    Validate that a numeric value is finite and within configured absolute bounds.

    Args:
        x: Numeric value to validate.

    Raises:
        _UnsafeExpression: If x is non-finite or exceeds allowed magnitude.
    """
    if isinstance(x, int):
        if abs(x) > _MAX_ABS_INT:
            raise _UnsafeExpression("Integer out of bounds")
    elif isinstance(x, float):
        if not math.isfinite(x) or abs(x) > _MAX_ABS_FLOAT:
            raise _UnsafeExpression("Float out of bounds")
    else:
        raise _UnsafeExpression("Non-numeric value where numeric required")


def _eval_node(node: ast.AST, depth: int) -> Union[int, float, bool]:
    """
    Recursively evaluate a whitelisted AST node.

    The evaluator strictly permits a small set of expression nodes and operators.
    It also enforces a maximum recursion depth to prevent pathologically deep
    expressions from consuming resources.

    Args:
        node: The AST node to evaluate.
        depth: Current recursion depth.

    Returns:
        int | float | bool: The evaluated result.

    Raises:
        _UnsafeExpression: On any disallowed node, operator, type mismatch,
                           or numeric bound violation.
    """
    if depth > _MAX_DEPTH:
        raise _UnsafeExpression("Expression too deep")

    # Constants (numbers and booleans only).
    if isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, bool):
            return val
        if isinstance(val, int):
            _check_number_bounds(val)
            return val
        if isinstance(val, float):
            _check_number_bounds(val)
            return val
        raise _UnsafeExpression("Unsupported literal type")

    # Legacy numeric nodes (for older Python ASTs).
    if isinstance(node, ast.Num):  # type: ignore[attr-defined]
        val = node.n  # type: ignore[attr-defined]
        if isinstance(val, (int, float)):
            _check_number_bounds(val)
            return val
        raise _UnsafeExpression("Unsupported numeric literal")

    # Disallow sequences and mapping literals to avoid composite results.
    if isinstance(node, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
        raise _UnsafeExpression("Sequences are not allowed")

    # Binary operations.
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS or op_type is ast.MatMult:
            raise _UnsafeExpression("Operator not allowed")

        left = _eval_node(node.left, depth + 1)
        right = _eval_node(node.right, depth + 1)

        # For exponentiation, limit both base and exponent magnitudes.
        if op_type is ast.Pow:
            if not _is_number(left) or not _is_number(right):
                raise _UnsafeExpression("Power requires numeric operands")
            if abs(float(left)) > _MAX_POW_BASE_ABS:
                raise _UnsafeExpression("Base too large for exponentiation")
            if abs(float(right)) > _MAX_POW_EXPONENT:
                raise _UnsafeExpression("Exponent too large")
        else:
            # Other numeric binops: operands must be numeric or boolean.
            if not (_is_number(left) or isinstance(left, bool)) or not (
                _is_number(right) or isinstance(right, bool)
            ):
                raise _UnsafeExpression("Operands must be numeric or boolean")

        # Compute result and validate bounds if numeric.
        try:
            res = _BIN_OPS[op_type](left, right)
        except Exception as exc:
            raise _UnsafeExpression(str(exc))
        if isinstance(res, (int, float)):
            _check_number_bounds(res)
        return res

    # Unary operations.
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        operand = _eval_node(node.operand, depth + 1)

        if op_type in _UNARY_NUMERIC:
            if not (_is_number(operand) or isinstance(operand, bool)):
                raise _UnsafeExpression("Unary operand must be numeric or boolean")
            res = _UNARY_NUMERIC[op_type](operand)
            if isinstance(res, (int, float)):
                _check_number_bounds(res)
            return res

        if op_type is ast.Not:
            if not isinstance(operand, bool):
                raise _UnsafeExpression("Logical not requires boolean")
            return not operand

        raise _UnsafeExpression("Unary operator not allowed")

    # Boolean operations (and/or) with strict boolean operands.
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            current = _eval_node(node.values[0], depth + 1)
            if not isinstance(current, bool):
                raise _UnsafeExpression("Boolean operations require booleans")
            for sub in node.values[1:]:
                if not current:
                    return False
                nxt = _eval_node(sub, depth + 1)
                if not isinstance(nxt, bool):
                    raise _UnsafeExpression("Boolean operations require booleans")
                current = current and nxt
            return current

        if isinstance(node.op, ast.Or):
            current = _eval_node(node.values[0], depth + 1)
            if not isinstance(current, bool):
                raise _UnsafeExpression("Boolean operations require booleans")
            for sub in node.values[1:]:
                if current:
                    return True
                nxt = _eval_node(sub, depth + 1)
                if not isinstance(nxt, bool):
                    raise _UnsafeExpression("Boolean operations require booleans")
                current = current or nxt
            return current

        raise _UnsafeExpression("Boolean operator not allowed")

    # Comparison chains such as a < b <= c.
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, depth + 1)
        comparators = [_eval_node(c, depth + 1) for c in node.comparators]
        ops = node.ops

        values = [left] + comparators
        for i, op in enumerate(ops):
            a = values[i]
            b = values[i + 1]

            if not (
                isinstance(a, (int, float, bool)) and isinstance(b, (int, float, bool))
            ):
                raise _UnsafeExpression(
                    "Comparisons require numeric or boolean values"
                )

            if not isinstance(op, _CMP_OPS):
                raise _UnsafeExpression("Comparison operator not allowed")

            if isinstance(op, ast.Eq):
                ok = a == b
            elif isinstance(op, ast.NotEq):
                ok = a != b
            elif isinstance(op, ast.Lt):
                ok = a < b
            elif isinstance(op, ast.LtE):
                ok = a <= b
            elif isinstance(op, ast.Gt):
                ok = a > b
            elif isinstance(op, ast.GtE):
                ok = a >= b
            else:
                # Defensive programming: the above checks should prevent this.
                raise _UnsafeExpression("Comparison operator not allowed")

            if not isinstance(ok, bool):
                raise _UnsafeExpression("Invalid comparison result")

            if not ok:
                return False

        return True

    # Conditional expression (ternary): a if cond else b
    if isinstance(node, ast.IfExp):
        test_val = _eval_node(node.test, depth + 1)
        if not isinstance(test_val, bool):
            raise _UnsafeExpression("Ternary condition must be boolean")
        if test_val:
            return _eval_node(node.body, depth + 1)
        return _eval_node(node.orelse, depth + 1)

    # Disallow names, calls, attributes, subscripts, comprehensions, statements, etc.
    if isinstance(node, _UNSAFE_NODE_TYPES):
        raise _UnsafeExpression("Unsupported or unsafe expression element")

    # If we encounter an AST node we didn't explicitly handle, reject it.
    raise _UnsafeExpression("Unsupported expression")
