"""
Safe mathematical expression evaluator.

This module exposes a single function, `perform_safe_math`, which evaluates a
user-provided mathematical expression string in a safe and restricted manner.
It parses the expression into an AST and evaluates only a whitelisted subset of
Python's arithmetic operations. Any attempt to use unsupported or unsafe
constructs will raise a ValueError.

Security constraints:
- Only numeric literals (ints and floats) are allowed.
- Boolean values are explicitly disallowed (True/False).
- Only arithmetic operations (+, -, *, /, //, %, **) and unary +/- are allowed.
- Name access, attribute access, function calls, comprehensions, etc. are all
  disallowed.
- Expression and result magnitudes are bounded to prevent excessive resource
  usage (e.g., extremely large integers or float overflows).

Logging:
- This module uses the standard logging library. A module-level logger is
  defined as `logger = logging.getLogger(__name__)`.
- On success, the function logs at INFO level: the raw expression and result.
- On failure (ValueError), the function logs at WARNING level: the raw
  expression and the error message.
- Unexpected errors are logged at ERROR level and re-raised as ValueError.
"""

import ast
import logging
import math
from typing import Union


# Module-level logger; applications should configure handlers/levels as needed.
logger = logging.getLogger(__name__)


# Simple numeric type alias for clarity.
Number = Union[int, float]


# Limits to keep evaluation safe and bounded.
# These limits aim to prevent resource exhaustion and numeric overflows while
# maintaining useful mathematical capabilities for most inputs.
_MAX_EXPR_LEN = 1000
_MAX_AST_NODES = 1000

# Maximum digits allowed in intermediate integer results. This prevents creation
# of unbounded large integers that could cause memory or performance issues.
_MAX_INT_DIGITS = 1000

# Roughly the maximum finite double value; used to bound float results.
_MAX_FLOAT_ABS = 1e308

# Constraints on exponentiation to prevent runaway growth.
_MAX_POW_INT_BASE_ABS = 1_000_000
_MAX_POW_INT_EXP_ABS = 100
_MAX_POW_FLOAT_BASE_ABS = 1_000_000.0
_MAX_POW_FLOAT_EXP_ABS = 1_000.0


def perform_safe_math(expression: str) -> Number:
    """
    Evaluate a mathematical expression string in a safe, controlled way.

    The evaluation supports:
      - Numeric constants: ints and floats
      - Unary operators: +, -
      - Binary operators: +, -, *, /, //, %, **

    All other constructs (names, attribute access, function calls, etc.) are
    rejected. Booleans are also rejected (even though they are ints in Python).

    Logging:
        On success, logs the raw input expression and the computed result at
        INFO level. On failure, logs the raw expression and the error at
        WARNING or ERROR level.

    Args:
        expression: A string containing the mathematical expression to evaluate.

    Returns:
        The numerical result (int or float) of the evaluated expression.

    Raises:
        ValueError: If the input is invalid, contains unsafe constructs, is too
            large or complex, or would result in an unsafe numeric value.
    """
    raw_expression = expression

    try:
        # Basic validation of the input type and size.
        if not isinstance(expression, str):
            raise ValueError("Expression must be a string.")

        expr = expression.strip()
        if not expr:
            raise ValueError("Expression cannot be empty or whitespace.")
        if len(expr) > _MAX_EXPR_LEN:
            raise ValueError("Expression is too long.")

        # Parse the expression into an AST in eval mode (single expression only).
        try:
            tree = ast.parse(expr, mode="eval")
        except Exception:
            # Hide Python-specific parse errors behind a consistent ValueError.
            raise ValueError("Invalid expression syntax.")

        # Limit expression complexity by bounding the number of AST nodes.
        if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
            raise ValueError("Expression is too complex.")

        # Whitelisted operations:
        allowed_bin_ops = (
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
        )
        allowed_unary_ops = (ast.UAdd, ast.USub)

        def _ensure_number(value: object) -> Number:
            """
            Ensure that a value is a numeric type (int or float), but not bool.

            Raises:
                ValueError: If the value is not an int/float or is a bool.
            """
            # bool is a subclass of int in Python; disallow explicitly.
            if isinstance(value, bool):
                raise ValueError("Booleans are not allowed.")
            if not isinstance(value, (int, float)):
                raise ValueError("Only numeric values are allowed.")
            return value

        def _check_limits(value: Number) -> Number:
            """
            Validate that a numeric result is within allowed magnitude limits.

            For ints, limits the number of digits. For floats, ensures finiteness
            and a reasonable absolute bound.

            Raises:
                ValueError: If the value exceeds configured bounds.
            """
            if isinstance(value, int):
                # Limit number of digits for big integers.
                if value == 0:
                    return value
                if len(str(abs(value))) > _MAX_INT_DIGITS:
                    raise ValueError("Result exceeds allowed integer size.")
                return value

            # For floats, ensure finiteness and magnitude bounds.
            if not math.isfinite(value):
                raise ValueError("Non-finite float result.")
            if abs(value) > _MAX_FLOAT_ABS:
                raise ValueError("Float result exceeds allowed magnitude.")
            return value

        def _safe_pow(left: Number, right: Number) -> Number:
            """
            Safely compute exponentiation under constrained bounds.

            - If both operands are ints, apply tight bounds on base and exponent.
            - If any operand is a float, apply separate bounds and guard overflow.
            """
            _ensure_number(left)
            _ensure_number(right)

            # If both are integers, constrain base and exponent tightly.
            if isinstance(left, int) and isinstance(right, int):
                if abs(left) > _MAX_POW_INT_BASE_ABS:
                    raise ValueError("Exponentiation base too large.")
                if abs(right) > _MAX_POW_INT_EXP_ABS:
                    raise ValueError("Exponent too large.")
                result_int = left ** right
                return _check_limits(result_int)

            # For any float involvement, keep base/exponent within reasonable bounds.
            if abs(float(left)) > _MAX_POW_FLOAT_BASE_ABS:
                raise ValueError("Exponentiation base too large.")
            if abs(float(right)) > _MAX_POW_FLOAT_EXP_ABS:
                raise ValueError("Exponent too large.")

            try:
                result_float = left ** right
            except OverflowError:
                raise ValueError("Exponentiation overflow.")
            return _check_limits(result_float)

        def _eval(node: ast.AST) -> Number:
            """
            Recursively and safely evaluate supported AST nodes.

            Supported nodes:
                - ast.Expression
                - ast.Constant (numeric only)
                - ast.Num (for older Python versions, numeric only)
                - ast.UnaryOp with allowed_unary_ops
                - ast.BinOp with allowed_bin_ops
            """
            # Root expression container in eval mode.
            if isinstance(node, ast.Expression):
                return _eval(node.body)

            # Numeric literals (Python 3.8+ use ast.Constant).
            if isinstance(node, ast.Constant):
                val = node.value
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise ValueError("Only numeric constants are allowed.")
                return _ensure_number(val)

            # Compatibility with older Python versions where ast.Num is used.
            if hasattr(ast, "Num") and isinstance(node, getattr(ast, "Num")):
                val = node.n
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise ValueError("Only numeric constants are allowed.")
                return _ensure_number(val)

            # Unary operations (+x, -x).
            if isinstance(node, ast.UnaryOp):
                if not isinstance(node.op, allowed_unary_ops):
                    raise ValueError("Unsupported unary operator.")
                operand = _ensure_number(_eval(node.operand))
                if isinstance(node.op, ast.UAdd):
                    return _check_limits(+operand)
                if isinstance(node.op, ast.USub):
                    return _check_limits(-operand)

            # Binary operations.
            if isinstance(node, ast.BinOp):
                if not isinstance(node.op, allowed_bin_ops):
                    raise ValueError("Unsupported binary operator.")
                left = _ensure_number(_eval(node.left))
                right = _ensure_number(_eval(node.right))

                if isinstance(node.op, ast.Add):
                    return _check_limits(left + right)
                if isinstance(node.op, ast.Sub):
                    return _check_limits(left - right)
                if isinstance(node.op, ast.Mult):
                    return _check_limits(left * right)
                if isinstance(node.op, ast.Div):
                    if right == 0:
                        raise ValueError("Division by zero.")
                    return _check_limits(left / right)
                if isinstance(node.op, ast.FloorDiv):
                    if right == 0:
                        raise ValueError("Division by zero.")
                    return _check_limits(left // right)
                if isinstance(node.op, ast.Mod):
                    if right == 0:
                        raise ValueError("Modulo by zero.")
                    return _check_limits(left % right)
                if isinstance(node.op, ast.Pow):
                    return _safe_pow(left, right)

            # Any other AST node type is considered unsafe/unsupported.
            raise ValueError("Unsupported or unsafe expression element encountered.")

        # Evaluate the parsed AST and return the validated result.
        result = _eval(tree)

        # Log success with the raw expression and computed result.
        logger.info(
            "perform_safe_math success: expression=%r, result=%r",
            raw_expression,
            result,
        )
        return result

    except ValueError as err:
        # Log known/expected failures at WARNING level and re-raise.
        logger.warning(
            "perform_safe_math failure: expression=%r, error=%s",
            raw_expression,
            err,
        )
        raise
    except Exception as err:  # Unexpected error path
        # Log unexpected errors at ERROR level and present a ValueError to caller.
        logger.error(
            "perform_safe_math unexpected error: expression=%r, error=%r",
            raw_expression,
            err,
        )
        raise ValueError("Unexpected error during evaluation.") from err
