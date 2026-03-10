"""
Utilities to safely evaluate user-provided mathematical expressions.

This module exposes a single function, `evaluate_user_expression`, which parses
and evaluates a mathematical expression using Python's AST (Abstract Syntax
Tree). Only a small, explicitly allowed subset of Python's expression syntax is
permitted, preventing execution of arbitrary code and unsafe operations.

Logging:
    A module-level logger is provided (named after this module). The function
    logs the raw input expression and the final evaluation result at INFO
    level. Error conditions (e.g., invalid characters or division by zero) are
    logged at WARNING level.

    Library-friendly behavior is preserved by attaching a NullHandler, so the
    host application can configure logging without duplicate handlers.
"""

import re
import ast
import logging
import operator as op


# Library-friendly logger (no output unless the host application configures it).
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def evaluate_user_expression(expression: str):
    """
    Safely evaluate a mathematical expression provided by the user.

    The expression is parsed into an AST, and only a restricted set of numeric
    literals and arithmetic operations is allowed. Any names, function calls,
    attribute access, subscriptions, or other Python constructs are rejected.

    Parameters:
        expression (str): The user's mathematical expression.

    Returns:
        int | float: The numeric result of evaluating the expression.

    Raises:
        ValueError: If the input is not a string, contains invalid characters,
            the expression cannot be parsed, uses unsupported operations,
            or attempts division by zero.

    Examples:
        >>> evaluate_user_expression("1 + 2 * 3")
        7
        >>> evaluate_user_expression("(2.5 - 0.5) * 4")
        8.0
        >>> evaluate_user_expression("2 ** 3")
        8
        >>> evaluate_user_expression("1 / 0")
        Traceback (most recent call last):
            ...
        ValueError: Division by zero.
    """
    # Log the raw expression before any processing for observability.
    logger.info("Evaluating expression: %s", expression)

    # Validate input type early to provide a clear error to callers.
    if not isinstance(expression, str):
        logger.warning(
            "Invalid expression type: %s (value=%r)",
            type(expression).__name__,
            expression,
        )
        raise ValueError("Expression must be a string.")

    # Trim surrounding whitespace and ensure the expression is not empty.
    expr = expression.strip()
    if not expr:
        logger.warning("Empty expression submitted.")
        raise ValueError("Empty expression.")

    # Character-level allowlist to rule out obviously dangerous inputs before
    # parsing. This allows:
    #   - digits 0-9
    #   - whitespace
    #   - parentheses: ( )
    #   - decimal point: .
    #   - arithmetic operators: + - * / % and exponentiation **
    #   - scientific notation markers: e E (as part of numbers)
    #
    # Anything else (letters beyond e/E, underscores, commas, etc.) is rejected.
    if not re.fullmatch(r"[0-9\s+\-*/%().eE]+", expr):
        logger.warning("Invalid characters detected in expression: %r", expr)
        raise ValueError("Invalid characters in expression.")

    # Parse the expression into an AST in 'eval' mode (single expression).
    try:
        parsed_ast = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        # Map parsing errors to ValueError as part of the public contract.
        logger.warning("Invalid expression syntax: %r", expr)
        raise ValueError("Invalid expression.") from exc

    # Define the operations we allow:
    # - Binary operators: +, -, *, /, %, **
    # - Unary operators: +, -
    # Note: We intentionally do not allow floor division (//), bitwise ops,
    # matrix multiplication (@), or any other operators.
    allowed_bin_ops = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Mod: op.mod,
        ast.Pow: op.pow,
    }
    allowed_unary_ops = {
        ast.UAdd: op.pos,
        ast.USub: op.neg,
    }

    def _eval(node):
        """
        Recursively evaluate a safe subset of AST nodes.

        Only numeric constants, unary operations, and binary operations declared
        in the allowlists above are permitted. Any other node types will result
        in a ValueError to prevent execution of arbitrary code.
        """
        # The root of a parsed expression wraps the actual body node.
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        # Literal numbers only: ints and floats are allowed.
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            logger.warning("Non-numeric constant encountered: %r", node.value)
            raise ValueError("Only numeric constants are allowed.")

        # Unary operations: +x or -x
        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in allowed_unary_ops:
                logger.warning("Unsupported unary operation: %s", type(node.op))
                raise ValueError("Unsupported unary operation.")
            return allowed_unary_ops[type(node.op)](_eval(node.operand))

        # Binary operations: x + y, x - y, x * y, x / y, x % y, x ** y
        if isinstance(node, ast.BinOp):
            if type(node.op) not in allowed_bin_ops:
                logger.warning("Unsupported binary operation: %s", type(node.op))
                raise ValueError("Unsupported binary operation.")
            left = _eval(node.left)
            right = _eval(node.right)
            try:
                return allowed_bin_ops[type(node.op)](left, right)
            except ZeroDivisionError as exc:
                # Normalize ZeroDivisionError to ValueError for callers.
                logger.warning("Division by zero in expression: %r", expr)
                raise ValueError("Division by zero.") from exc

        # Disallow everything else: names, calls, attributes, subscripts,
        # lambdas, comprehensions, and control-flow constructs.
        logger.warning("Disallowed syntax encountered in AST: %s", type(node))
        raise ValueError("Invalid expression.")

    # Kick off the recursive evaluation from the parsed AST.
    result = _eval(parsed_ast)

    # Log the final result in a human-readable form.
    logger.info("Evaluation result: %s", result)

    return result
