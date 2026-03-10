"""Secure arithmetic expression parser and evaluator.

This module exposes a single public function, `parse_and_calculate`, which parses
and safely evaluates simple arithmetic expressions supplied by end users.

Safety is enforced in two stages:
1) A character-level allowlist regular expression that limits inputs to digits,
   whitespace, arithmetic operators, parentheses, and the decimal point.
2) An AST-level validator/evaluator that only permits numeric constants and the
   standard arithmetic operators (+, -, *, /, //, %, **), along with unary +/-.
"""

import re
import ast


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


def parse_and_calculate(expression: str):
    """Parse and calculate a simple arithmetic expression securely.

    This function is intended for a text-based calculator that supports basic
    arithmetic on integers and floating-point numbers. It rejects inputs that
    contain unsupported characters or AST nodes that could be unsafe.

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

    # 1) Character-level validation (fast allowlist).
    # This rejects inputs with letters, underscores, quotes, commas, etc.
    if not _ALLOWED_TOKENS_RE.match(expression):
        raise ValueError("Expression contains unsupported characters.")

    # 2) Parse into an AST and validate structure during evaluation.
    try:
        ast_tree = ast.parse(expression, mode="eval")
    except Exception:
        # Any parsing failure is reported as invalid syntax to the caller.
        raise ValueError("Invalid expression syntax.")

    # Evaluate the validated AST. Catch common arithmetic issues and rephrase.
    try:
        return _eval_ast(ast_tree)
    except ZeroDivisionError:
        raise ValueError("Division by zero.")
    except OverflowError:
        raise ValueError("Computation overflow.")
    except ValueError:
        # Re-raise exact ValueErrors from validation/evaluation.
        raise
    except Exception:
        # Any other unexpected evaluation errors.
        raise ValueError("Cannot compute expression.")


def _eval_ast(node):
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

    Returns:
        The numeric result of evaluating the node.

    Raises:
        ValueError: If the node type or operator is not allowed.
    """
    # Unwrap the top-level expression.
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)

    # Numeric constants.
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Unsupported constant type.")

    # Python < 3.8 compatibility: ast.Num was the numeric node type.
    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
        return node.n

    # Unary operations: +x, -x
    if isinstance(node, ast.UnaryOp):
        operand_value = _eval_ast(node.operand)

        if isinstance(node.op, ast.UAdd):
            return +operand_value
        if isinstance(node.op, ast.USub):
            return -operand_value

        raise ValueError("Unsupported unary operator.")

    # Binary operations: x op y
    if isinstance(node, ast.BinOp):
        left_value = _eval_ast(node.left)
        right_value = _eval_ast(node.right)

        if isinstance(node.op, ast.Add):
            return left_value + right_value
        if isinstance(node.op, ast.Sub):
            return left_value - right_value
        if isinstance(node.op, ast.Mult):
            return left_value * right_value
        if isinstance(node.op, ast.Div):
            return left_value / right_value
        if isinstance(node.op, ast.FloorDiv):
            return left_value // right_value
        if isinstance(node.op, ast.Mod):
            return left_value % right_value
        if isinstance(node.op, ast.Pow):
            # Guard against extremely large exponents that may hang or overflow.
            if isinstance(right_value, (int, float)) and abs(right_value) > 10000:
                raise ValueError("Exponent too large.")
            return left_value ** right_value

        raise ValueError("Unsupported binary operator.")

    # Any other AST node (e.g., Name, Call, Attribute, comprehensions, etc.) is disallowed.
    raise ValueError("Unsupported or potentially malicious expression structure.")
