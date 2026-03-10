"""
A safe evaluator for simple arithmetic expressions written in Python syntax.

This module parses a user-supplied expression into an AST, validates that the
AST only contains a tightly controlled subset of safe numeric operations, and
then evaluates it without using Python's eval or executing arbitrary code.

Only numeric constants (ints and floats) and the following operators are
supported:
- Unary: +, -
- Binary: +, -, *, /, //, %, **

Any unsupported syntax or evaluation error results in "Computation Error!".
"""

import ast
import math
from typing import Union

# -----------------------------------------------------------------------------
# Safety configuration
# -----------------------------------------------------------------------------

# Maximum length of the input expression string.
_MAX_EXPR_LENGTH = 1000

# Maximum number of AST nodes allowed (prevents excessively large/complex input).
_MAX_AST_NODES = 1000

# Maximum recursion depth during AST evaluation (prevents pathological cases).
_MAX_RECURSION_DEPTH = 200

# Absolute value limit for exponent in power operations (guards huge exponentiation).
_MAX_POWER_EXPONENT = 10000

# Canonical error message returned on any validation or computation failure.
_ERROR_MSG = "Computation Error!"

# Type alias for supported numeric results.
_Number = Union[int, float]


# -----------------------------------------------------------------------------
# AST validation helpers
# -----------------------------------------------------------------------------
def _is_allowed_constant(value: object) -> bool:
    """
    Determine if a literal constant value is allowed.

    We only permit plain ints and floats (bool is disallowed even though it is
    a subclass of int). All other constant types (str, bytes, complex, etc.)
    are rejected.

    Args:
        value: The constant value to validate.

    Returns:
        True if the value is an allowed numeric type; False otherwise.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    return False


def _is_safe_ast(tree: ast.AST) -> bool:
    """
    Check that the parsed AST contains only a safe subset of nodes and operators.

    Allowed:
    - ast.Expression (root)
    - ast.Constant with numeric values (ints/floats; bools rejected)
    - ast.UnaryOp with ast.UAdd or ast.USub
    - ast.BinOp with ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow

    Any other node type (including ast.Name, ast.Call, attributes, comprehensions,
    lambdas, etc.) is considered unsafe.

    Args:
        tree: The AST parsed from the input expression.

    Returns:
        True if the AST is safe; False otherwise.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Expression):
            # Root container for eval-mode parsing.
            continue

        if isinstance(node, ast.BinOp):
            if not isinstance(
                node.op,
                (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow),
            ):
                return False
            continue

        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub)):
                return False
            continue

        if isinstance(node, ast.Constant):
            if not _is_allowed_constant(node.value):
                return False
            continue

        # ast.Load is a harmless context marker, but if we encounter any "Name"
        # or other node types, we'll reject them below. We allow Load specifically
        # so its presence does not cause a false negative.
        if isinstance(node, ast.Load):
            continue

        # Any other node is not allowed.
        return False

    return True


# -----------------------------------------------------------------------------
# AST evaluation
# -----------------------------------------------------------------------------
def _eval_ast(node: ast.AST, depth: int = 0) -> _Number:
    """
    Recursively evaluate a validated AST node representing an arithmetic expression.

    This function assumes the AST has already been validated by _is_safe_ast.
    It still performs type checks and guards to avoid numeric explosions and
    other runtime issues.

    Args:
        node: The AST node to evaluate.
        depth: The current recursion depth (used to enforce limits).

    Returns:
        The numeric result of evaluating the node.

    Raises:
        ValueError: If an unsupported construct is encountered or limits are exceeded.
    """
    if depth > _MAX_RECURSION_DEPTH:
        raise ValueError("Too deep")

    if isinstance(node, ast.Constant):
        value = node.value
        if not _is_allowed_constant(value):
            raise ValueError("Disallowed constant")
        return value  # type: ignore[return-value]

    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand, depth + 1)
        if not isinstance(operand, (int, float)):
            raise ValueError("Non-numeric operand")

        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand

        raise ValueError("Disallowed unary operator")

    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left, depth + 1)
        right = _eval_ast(node.right, depth + 1)

        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            raise ValueError("Non-numeric operand")

        op = node.op
        try:
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.Div):
                return left / right
            if isinstance(op, ast.FloorDiv):
                return left // right
            if isinstance(op, ast.Mod):
                return left % right
            if isinstance(op, ast.Pow):
                # Guard against massive exponentiation results.
                if isinstance(right, (int, float)) and abs(right) > _MAX_POWER_EXPONENT:
                    raise ValueError("Exponent too large")
                return left ** right
        except Exception as exc:
            # Normalize any runtime error (e.g., ZeroDivisionError) into ValueError
            # so the top-level function can map it to the canonical error string.
            raise ValueError(str(exc)) from exc

        raise ValueError("Disallowed binary operator")

    if isinstance(node, ast.Expression):
        # Evaluate the expression body.
        return _eval_ast(node.body, depth + 1)

    # Any other node is not allowed.
    raise ValueError("Disallowed syntax")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def evaluate_expression(expr_string: str) -> str:
    """
    Evaluate a basic arithmetic expression represented as a Python-syntax string.

    The input is parsed into an AST, validated for safety, and evaluated using
    a small, controlled interpreter (no eval/exec). Only numeric literals
    and a limited set of arithmetic operators are supported.

    Args:
        expr_string: The user-supplied expression string.

    Returns:
        The computed result as a string, or "Computation Error!" if validation
        fails, the expression is unsafe, or evaluation encounters an error.
    """
    try:
        # Basic input validation.
        if not isinstance(expr_string, str):
            return _ERROR_MSG

        expr = expr_string.strip()
        if not expr:
            return _ERROR_MSG

        if len(expr) > _MAX_EXPR_LENGTH:
            return _ERROR_MSG

        # Parse the expression in evaluation mode to get a single Expression node.
        tree = ast.parse(expr, mode="eval")

        # Structural complexity guard.
        if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
            return _ERROR_MSG

        # Validate AST for safety.
        if not _is_safe_ast(tree):
            return _ERROR_MSG

        # Evaluate safely.
        result = _eval_ast(tree.body, 0)

        # Ensure the result is numeric and not NaN/Inf.
        if isinstance(result, float) and not math.isfinite(result):
            return _ERROR_MSG
        if not isinstance(result, (int, float)):
            return _ERROR_MSG

        return str(result)
    except Exception:
        # Any parsing/validation/evaluation error maps to the canonical error.
        return _ERROR_MSG
