"""
A safe arithmetic expression evaluator with structured logging.

This module provides:
- A guarded, AST-based evaluator that supports basic arithmetic operations.
- Guard clauses to validate input early and fail fast on unsafe or invalid expressions.
- Structured JSON logging of each evaluation attempt (both success and error) to a rotating log file.
"""

import re
import ast
import logging
import json
from logging.handlers import RotatingFileHandler
from typing import Any, Union

# Configure structured logger
_LOGGER = logging.getLogger("calculator.expression")
if not _LOGGER.handlers:
    _LOGGER.setLevel(logging.INFO)
    _handler = RotatingFileHandler("calculator.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    _handler.setFormatter(
        logging.Formatter(
            fmt='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":%(message)s}',
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    _LOGGER.addHandler(_handler)
    _LOGGER.propagate = False


def _log_json(event: str, **fields: Any) -> str:
    """
    Build a JSON string for structured logging.

    Args:
        event: A short event name (e.g., 'evaluate_expression_success').
        **fields: Arbitrary key-value pairs to include in the payload.

    Returns:
        A JSON string to be used as the log message.
    """
    payload = {"event": event}
    payload.update(fields)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


_ALLOWED_CHARS_RE = re.compile(r'^[\s0-9+\-*/%()._]+$')


def _safe_div(a: Union[int, float], b: Union[int, float]) -> float:
    """
    Safely perform true division, guarding against division by zero.

    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        raise ValueError("Division by zero.")
    return a / b


def _safe_floordiv(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """
    Safely perform floor division, guarding against division by zero.

    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        raise ValueError("Division by zero.")
    return a // b


def _safe_mod(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """
    Safely compute modulo, guarding against modulo by zero.

    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        raise ValueError("Modulo by zero.")
    return a % b


_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: _safe_div,
    ast.FloorDiv: _safe_floordiv,
    ast.Mod: _safe_mod,
    ast.Pow: lambda a, b: a ** b,
}

_UNARY_OPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


class _SafeEvaluator(ast.NodeVisitor):
    """
    An AST visitor that safely evaluates arithmetic expressions.

    Supported nodes:
    - Expression
    - BinOp with operators: +, -, *, /, //, %, **
    - UnaryOp with operators: +, -
    - Numeric literals (int, float)

    Any other node type will raise ValueError.
    """

    def generic_visit(self, node):
        """Reject any unsupported AST node by default."""
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    def visit_Expression(self, node: ast.Expression):
        """Evaluate the root Expression node by visiting its body."""
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp):
        """Evaluate a binary operation after validating operand types and operator support."""
        left = self.visit(node.left)
        right = self.visit(node.right)

        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            raise ValueError("Invalid operands in expression.")

        op_func = _BIN_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")

        result = op_func(left, right)
        if isinstance(result, complex):
            raise ValueError("Complex numbers are not supported.")
        return result

    def visit_UnaryOp(self, node: ast.UnaryOp):
        """Evaluate a unary operation after validating the operand and operator."""
        operand = self.visit(node.operand)
        if not isinstance(operand, (int, float)):
            raise ValueError("Invalid operand in unary operation.")

        op_func = _UNARY_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

        return op_func(operand)

    # Python <3.8 numeric literal
    def visit_Num(self, node: ast.Num):
        """Validate and return a numeric literal (legacy for Python <3.8)."""
        value = node.n
        if isinstance(value, bool):
            raise ValueError("Booleans are not supported.")
        if not isinstance(value, (int, float)):
            raise ValueError(f"Unsupported numeric literal: {type(value).__name__}")
        return value

    # Python 3.8+ constant
    def visit_Constant(self, node: ast.Constant):
        """Validate and return a constant (int or float only)."""
        value = node.value
        if isinstance(value, bool):
            raise ValueError("Booleans are not supported.")
        if not isinstance(value, (int, float)):
            raise ValueError(f"Unsupported constant: {type(value).__name__}")
        return value


def evaluate_expression(expr: str):
    """
    Safely evaluate a simple mathematical expression.

    The function:
    - Validates input type and characters (guard clauses).
    - Parses the expression using Python's AST in 'eval' mode.
    - Evaluates only a restricted subset of AST nodes and operators.
    - Logs both the incoming expression and its final outcome (success or error)
      in structured JSON lines to 'calculator.log'.

    Args:
        expr: A string representing the arithmetic expression provided by the user.

    Returns:
        The computed numeric result of the expression (int or float).

    Raises:
        ValueError: If unsupported characters, unsafe commands, or invalid operations
                    are detected (e.g., division by zero), or if parsing fails.
    """
    # Type check
    if not isinstance(expr, str):
        _LOGGER.error(_log_json(
            "evaluate_expression_error",
            status="error",
            error="Expression must be a string.",
            expr_type=type(expr).__name__,
        ))
        raise ValueError("Expression must be a string.")

    original_expr = expr
    expr = expr.strip()

    # Empty expression
    if not expr:
        _LOGGER.error(_log_json(
            "evaluate_expression_error",
            status="error",
            expr=original_expr,
            error="Empty expression."
        ))
        raise ValueError("Empty expression.")

    # Unsupported characters (only digits, whitespace, parentheses, and basic operators)
    if not _ALLOWED_CHARS_RE.fullmatch(expr):
        _LOGGER.error(_log_json(
            "evaluate_expression_error",
            status="error",
            expr=expr,
            error="Unsupported characters detected in expression."
        ))
        raise ValueError("Unsupported characters detected in expression.")

    # Parse to AST
    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError:
        _LOGGER.error(_log_json(
            "evaluate_expression_error",
            status="error",
            expr=expr,
            error="Invalid expression syntax."
        ))
        raise ValueError("Invalid expression.")

    evaluator = _SafeEvaluator()

    # Evaluate AST with error handling
    try:
        result = evaluator.visit(parsed)
    except ValueError as e:
        _LOGGER.error(_log_json(
            "evaluate_expression_error",
            status="error",
            expr=expr,
            error=str(e)
        ))
        raise
    except Exception as e:
        _LOGGER.error(_log_json(
            "evaluate_expression_error",
            status="error",
            expr=expr,
            error=f"Unexpected error: {e.__class__.__name__}"
        ))
        raise ValueError("Invalid expression.")

    # Normalize negative zero to positive zero for consistency
    if isinstance(result, float) and result == 0.0:
        result = 0.0

    # Success log
    _LOGGER.info(_log_json(
        "evaluate_expression_success",
        status="success",
        expr=expr,
        result=result
    ))

    return result
