"""Safe mathematical expression evaluator.

This module provides a single function, `safe_eval_expression`, which safely
evaluates a limited set of mathematical expressions provided by end users.
It uses Python's `ast` module to parse and strictly validate the expression
before evaluating it with a small, controlled interpreter. This is intended
for scenarios such as web-based calculators.

Security model:
- Only digits, decimal points, basic arithmetic operators, whitespace, and
  parentheses are allowed at the character level.
- The parsed AST is validated to contain only numeric literals and arithmetic
  operations (no names, calls, attributes, or other side-effectful nodes).
"""

import ast
import re
from typing import Union

# A numeric type alias for function annotations and clarity.
Number = Union[int, float]

# Character-level validation pattern.
# This regex ensures the input contains ONLY:
# - digits: \d
# - whitespace: \s
# - parentheses: ( )
# - decimal point: .
# - arithmetic operators: + - * / %
# Note:
# - Exponentiation (**) is allowed because it's composed of two '*' characters,
#   both of which are permitted by this character-level check. The AST
#   validation below explicitly allows Pow as an operator as well.
_ALLOWED_CHAR_PATTERN = re.compile(r"^[\d\s\+\-\*\/\%\(\)\.]+$")

# Whitelist of AST node types that are allowed in the parsed expression.
# Keeping this at module scope avoids reconstructing the tuple on each call.
ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Num,       # For Python < 3.8
    ast.Constant,  # For Python >= 3.8
    ast.Load,
    # Binary operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    # Unary operators
    ast.UAdd,
    ast.USub,
    # Expression wrapper
    ast.Expr,
)


def safe_eval_expression(expr: str) -> Number:
    """Safely evaluate a mathematical expression string.

    The evaluation is constrained to numeric literals and basic arithmetic
    operations. If the input contains invalid characters or parses to an
    unsupported/unsafe AST, a ValueError is raised.

    Args:
        expr: The user's mathematical expression as a string.

    Returns:
        The numeric result of evaluating the expression.

    Raises:
        ValueError: If the input contains invalid characters, cannot be parsed,
            includes unsupported AST nodes, attempts invalid operations, or
            otherwise forms an incorrect expression.
    """
    # Basic validation: must be a non-empty string.
    if not isinstance(expr, str):
        raise ValueError("Expression must be a string")
    if not expr.strip():
        raise ValueError("Expression is empty")

    # Character-level allowlist check to reject anything suspicious early
    # (e.g., letters, underscores, quotes, brackets other than parentheses).
    if not _ALLOWED_CHAR_PATTERN.fullmatch(expr):
        raise ValueError("Invalid characters in expression")

    # Parse to an AST in 'eval' mode (single expression, not statements).
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Incorrect expression") from exc

    # Walk the AST to ensure only permitted node types are present and that
    # constants are numeric (ints or floats).
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_AST_NODES):
            raise ValueError("Invalid expression")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            # Reject non-numeric constants (e.g., strings, None, booleans).
            # Note: bool is a subclass of int, so this check also excludes bool.
            if isinstance(node.value, bool):
                raise ValueError("Invalid constant in expression")
            raise ValueError("Invalid constant in expression")

    def eval_node(ast_node: ast.AST) -> Number:
        """Recursively and safely evaluate permitted AST nodes."""
        if isinstance(ast_node, ast.Expression):
            return eval_node(ast_node.body)

        if isinstance(ast_node, ast.Num):
            return ast_node.n

        if isinstance(ast_node, ast.Constant):
            value = ast_node.value
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
            raise ValueError("Invalid constant in expression")

        if isinstance(ast_node, ast.UnaryOp):
            operand = eval_node(ast_node.operand)
            if isinstance(ast_node.op, ast.UAdd):
                return +operand
            if isinstance(ast_node.op, ast.USub):
                return -operand
            raise ValueError("Invalid unary operator")

        if isinstance(ast_node, ast.BinOp):
            left = eval_node(ast_node.left)
            right = eval_node(ast_node.right)
            operator = ast_node.op
            try:
                if isinstance(operator, ast.Add):
                    return left + right
                if isinstance(operator, ast.Sub):
                    return left - right
                if isinstance(operator, ast.Mult):
                    return left * right
                if isinstance(operator, ast.Div):
                    return left / right
                if isinstance(operator, ast.FloorDiv):
                    return left // right
                if isinstance(operator, ast.Mod):
                    return left % right
                if isinstance(operator, ast.Pow):
                    return left ** right
            except Exception as exc:
                # Normalize runtime errors to ValueError per spec.
                raise ValueError("Incorrect expression") from exc
            raise ValueError("Invalid operator")

        # Any other node reaching here is disallowed.
        raise ValueError("Invalid expression")

    try:
        return eval_node(tree)
    except ZeroDivisionError as exc:
        raise ValueError("Division by zero") from exc
    except RecursionError as exc:
        raise ValueError("Expression too complex") from exc
