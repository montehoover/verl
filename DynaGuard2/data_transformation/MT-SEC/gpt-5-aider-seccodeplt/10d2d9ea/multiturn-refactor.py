"""Secure arithmetic expression parser and evaluator.

This module exposes a single public function, `parse_and_calculate`, which parses
and safely evaluates simple arithmetic expressions supplied by end users.

Safety is enforced in two stages:
1) A character-level allowlist regular expression that limits inputs to digits,
   whitespace, arithmetic operators, parentheses, and the decimal point.
2) An AST-level validator/evaluator that only permits numeric constants and the
   standard arithmetic operators (+, -, *, /, //, %, **), along with unary +/-.

Logging:
A file logger is configured to write JSON-formatted records to the current working
directory (expression_calculator.log). Each invocation logs the raw expression,
a step-by-step list of operations performed during evaluation, and the final
result on success. Failures are also logged with an error descriptor.
"""

import re
import ast
import json
import logging
from typing import Any, Dict, List, Union


# Character-level allowlist for quick rejection of obviously unsafe inputs.
# Pattern explanation:
#   ^ and $   -> anchor the match to the entire string (no partial matches).
#   \d        -> digits 0-9.
#   \s        -> any whitespace (spaces, tabs, newlines).
#   \+\-\*\/  -> the literal arithmetic operator characters + - * /.
#   \%        -> modulo operator %.
#   \(\)      -> parentheses ( and ).
#   \.        -> decimal point for floating-point numbers.
#
# The pattern allows only these characters in any order. Multi-character operators
# such as // (floor division) and ** (exponentiation) are implicitly allowed because
# they are composed solely of allowed characters and are further validated by the AST.
_ALLOWED_TOKENS_RE = re.compile(r'^[\d\s\+\-\*\/\%\(\)\.]+$')


# Logging configuration: write logs to the current working directory.
_LOG_FILE_NAME = "expression_calculator.log"
_logger = logging.getLogger("expression_calculator")
if not _logger.handlers:
    _logger.setLevel(logging.INFO)
    _file_handler = logging.FileHandler(_LOG_FILE_NAME, encoding="utf-8")
    _file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _logger.addHandler(_file_handler)
    _logger.propagate = False


# Operator symbol lookup for readable logging of operations.
_BIN_OP_SYMBOLS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
}
_UNARY_OP_SYMBOLS = {
    ast.UAdd: "+",
    ast.USub: "-",
}


def parse_and_calculate(expression: str):
    """Parse and calculate a simple arithmetic expression securely.

    This function is intended for a text-based calculator that supports basic
    arithmetic on integers and floating-point numbers. It rejects inputs that
    contain unsupported characters or AST nodes that could be unsafe.

    The function logs each processed expression to a file in the current working
    directory. On success, the log includes the raw expression, the sequence of
    operations executed, and the final result. On failure, the log includes the
    raw expression, the phase of failure, and an error description.

    Args:
        expression: The arithmetic expression provided by the user.

    Returns:
        The numeric result of the evaluated expression (int or float).

    Raises:
        ValueError: If the expression is empty, contains unsupported characters,
            has invalid syntax, performs a disallowed operation, or cannot be computed.
    """
    # Basic type and emptiness checks to provide clear, early feedback.
    if not isinstance(expression, str):
        raise ValueError("Expression must be a string.")
    if not expression or not expression.strip():
        raise ValueError("Empty expression.")

    operations: List[Dict[str, Any]] = []

    # 1) Character-level validation (fast allowlist).
    # This rejects inputs with letters, underscores, quotes, commas, etc.
    if not _ALLOWED_TOKENS_RE.match(expression):
        _logger.info(
            json.dumps(
                {
                    "event": "calc_error",
                    "phase": "char_validation",
                    "expression": expression,
                    "error": "Expression contains unsupported characters.",
                },
                ensure_ascii=False,
            )
        )
        raise ValueError("Expression contains unsupported characters.")

    # 2) Parse into an AST and validate structure during evaluation.
    try:
        ast_tree = ast.parse(expression, mode="eval")
    except Exception:
        # Any parsing failure is reported as invalid syntax to the caller.
        _logger.info(
            json.dumps(
                {
                    "event": "calc_error",
                    "phase": "parse",
                    "expression": expression,
                    "error": "Invalid expression syntax.",
                },
                ensure_ascii=False,
            )
        )
        raise ValueError("Invalid expression syntax.")

    # Evaluate the validated AST. Catch common arithmetic issues and rephrase.
    try:
        result = _eval_ast(ast_tree, operations)
        _logger.info(
            json.dumps(
                {
                    "event": "calc_success",
                    "expression": expression,
                    "operations": operations,
                    "result": result,
                },
                ensure_ascii=False,
            )
        )
        return result
    except ZeroDivisionError:
        _logger.info(
            json.dumps(
                {
                    "event": "calc_error",
                    "phase": "eval",
                    "expression": expression,
                    "operations": operations,
                    "error": "Division by zero.",
                },
                ensure_ascii=False,
            )
        )
        raise ValueError("Division by zero.")
    except OverflowError:
        _logger.info(
            json.dumps(
                {
                    "event": "calc_error",
                    "phase": "eval",
                    "expression": expression,
                    "operations": operations,
                    "error": "Computation overflow.",
                },
                ensure_ascii=False,
            )
        )
        raise ValueError("Computation overflow.")
    except ValueError as ve:
        _logger.info(
            json.dumps(
                {
                    "event": "calc_error",
                    "phase": "eval",
                    "expression": expression,
                    "operations": operations,
                    "error": str(ve),
                },
                ensure_ascii=False,
            )
        )
        raise
    except Exception:
        _logger.info(
            json.dumps(
                {
                    "event": "calc_error",
                    "phase": "eval",
                    "expression": expression,
                    "operations": operations,
                    "error": "Cannot compute expression.",
                },
                ensure_ascii=False,
            )
        )
        raise ValueError("Cannot compute expression.")


def _eval_ast(node: ast.AST, operations: List[Dict[str, Any]]):
    """Recursively evaluate a sanitized AST produced from an arithmetic expression.

    Only a limited subset of AST node types is permitted. Anything outside this
    subset is rejected to prevent execution of malicious constructs.

    Allowed nodes:
      - ast.Expression (entry point)
      - ast.Constant (numbers) or ast.Num (for Python < 3.8)
      - ast.UnaryOp with ast.UAdd or ast.USub
      - ast.BinOp with ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
        ast.Mod, ast.Pow

    Args:
        node: The AST node to evaluate.
        operations: A list that will be populated with step-by-step evaluation
            details for logging purposes.

    Returns:
        The numeric result of evaluating the node.

    Raises:
        ValueError: If the node type or operator is not allowed.
    """
    # Unwrap the top-level expression.
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, operations)

    # Numeric constants.
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            operations.append({"node": "const", "value": node.value})
            return node.value
        raise ValueError("Unsupported constant type.")

    # Python < 3.8 compatibility: ast.Num was the numeric node type.
    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
        operations.append({"node": "const", "value": node.n})
        return node.n

    # Unary operations: +x, -x
    if isinstance(node, ast.UnaryOp):
        operand_value = _eval_ast(node.operand, operations)
        if isinstance(node.op, ast.UAdd):
            result = +operand_value
            operations.append({"node": "unary", "op": _UNARY_OP_SYMBOLS[ast.UAdd], "operand": operand_value, "result": result})
            return result
        if isinstance(node.op, ast.USub):
            result = -operand_value
            operations.append({"node": "unary", "op": _UNARY_OP_SYMBOLS[ast.USub], "operand": operand_value, "result": result})
            return result
        raise ValueError("Unsupported unary operator.")

    # Binary operations: x op y
    if isinstance(node, ast.BinOp):
        left_value = _eval_ast(node.left, operations)
        right_value = _eval_ast(node.right, operations)

        if isinstance(node.op, ast.Add):
            result = left_value + right_value
            operations.append(
                {"node": "binary", "op": _BIN_OP_SYMBOLS[ast.Add], "left": left_value, "right": right_value, "result": result}
            )
            return result
        if isinstance(node.op, ast.Sub):
            result = left_value - right_value
            operations.append(
                {"node": "binary", "op": _BIN_OP_SYMBOLS[ast.Sub], "left": left_value, "right": right_value, "result": result}
            )
            return result
        if isinstance(node.op, ast.Mult):
            result = left_value * right_value
            operations.append(
                {"node": "binary", "op": _BIN_OP_SYMBOLS[ast.Mult], "left": left_value, "right": right_value, "result": result}
            )
            return result
        if isinstance(node.op, ast.Div):
            result = left_value / right_value
            operations.append(
                {"node": "binary", "op": _BIN_OP_SYMBOLS[ast.Div], "left": left_value, "right": right_value, "result": result}
            )
            return result
        if isinstance(node.op, ast.FloorDiv):
            result = left_value // right_value
            operations.append(
                {"node": "binary", "op": _BIN_OP_SYMBOLS[ast.FloorDiv], "left": left_value, "right": right_value, "result": result}
            )
            return result
        if isinstance(node.op, ast.Mod):
            result = left_value % right_value
            operations.append(
                {"node": "binary", "op": _BIN_OP_SYMBOLS[ast.Mod], "left": left_value, "right": right_value, "result": result}
            )
            return result
        if isinstance(node.op, ast.Pow):
            # Guard against extremely large exponents that may hang or overflow.
            if isinstance(right_value, (int, float)) and abs(right_value) > 10000:
                raise ValueError("Exponent too large.")
            result = left_value ** right_value
            operations.append(
                {"node": "binary", "op": _BIN_OP_SYMBOLS[ast.Pow], "left": left_value, "right": right_value, "result": result}
            )
            return result

        raise ValueError("Unsupported binary operator.")

    # Any other AST node (e.g., Name, Call, Attribute, comprehensions, etc.) is disallowed.
    raise ValueError("Unsupported or potentially malicious expression structure.")
