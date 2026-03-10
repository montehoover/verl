"""
A safe mathematical expression evaluator used in a web-based calculator.

This module exposes a single public function, `process_user_query`, which takes
a user-supplied math expression string, validates it for safety, parses it into
an abstract syntax tree (AST), and evaluates only a restricted set of numeric
operations. Any unsafe content or malformed expressions result in a ValueError.

Logging:
    The module logs each received query and, on success, the computed result.
    Logs are human-readable and include timestamps and logger names. By default,
    a simple stream handler is attached if no handlers are configured for this
    module's logger.
"""

import ast
import logging
import re

# Configure module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    # Prevent double logging if root logger has handlers
    logger.propagate = False

# Precompiled regex for fast validation of allowed characters.
# Allowed characters:
# - digits: 0-9
# - whitespace: \s
# - decimal point: .
# - arithmetic operators: + - * / % ( ) with support for // and ** via AST
ALLOWED_CHARS_RE = re.compile(r'^[0-9\s\.\+\-\*\/\%\(\)]+$')


def process_user_query(query: str):
    """
    Process a user-provided mathematical expression and compute its result.

    The function performs the following steps:
    1) Validates that the input is a non-empty string containing only safe
       characters.
    2) Parses the expression into an AST in 'eval' mode.
    3) Recursively evaluates the AST, allowing only safe numeric operations.
    4) Logs the raw query and, on success, the computed result.

    Supported operators:
      - Binary: +, -, *, /, //, %, **
      - Unary: +, -
      - Parentheses: ( )

    Args:
        query (str): The mathematical expression to evaluate.

    Returns:
        The computed numeric result of the expression.

    Raises:
        ValueError: If the input contains unsafe characters, or the expression
                    is malformed (including division by zero).
    """
    # Log the receipt of the query early to ensure every attempt is tracked.
    if not isinstance(query, str):
        logger.info("Received query (non-string): %r", query)
        logger.warning(
            "Rejected query: non-string input type (%s)",
            type(query).__name__,
        )
        raise ValueError("Expression must be a string.")

    raw_query = query
    logger.info("Received query: %s", raw_query)

    expr = raw_query.strip()
    if not expr:
        logger.warning(
            "Rejected query: empty after trimming whitespace. Raw input: %r",
            raw_query,
        )
        raise ValueError("Malformed expression: empty input.")

    # Character-level safety validation
    if not ALLOWED_CHARS_RE.fullmatch(expr):
        logger.warning("Rejected query: unsafe characters. Raw input: %r", raw_query)
        raise ValueError("Unsafe characters in expression.")

    # Parse into an AST. Any syntax issues are treated as malformed expressions.
    try:
        parsed = ast.parse(expr, mode='eval')
    except SyntaxError:
        logger.warning("Malformed expression during parse. Raw input: %r", raw_query)
        raise ValueError("Malformed expression.")

    def eval_node(node):
        """
        Safely evaluate a subset of Python AST nodes representing numeric math.

        Permitted nodes:
          - Expression
          - BinOp with operators: Add, Sub, Mult, Div, FloorDiv, Mod, Pow
          - UnaryOp with operators: UAdd, USub
          - Numeric constants (int, float)

        Any other node type (e.g., Name, Call, Attribute, etc.) is rejected.
        """
        # Root expression wrapper
        if isinstance(node, ast.Expression):
            return eval_node(node.body)

        # Binary operations
        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)

            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left ** right

            # Any other operator is disallowed
            raise ValueError("Unsafe operator in expression.")

        # Unary operations (e.g., +3, -5)
        if isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)

            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand

            raise ValueError("Unsafe unary operator in expression.")

        # Numeric literals (Python 3.8+: ast.Constant)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Unsafe constant in expression.")

        # Backward compatibility for older Python versions (pre-3.8)
        if hasattr(ast, "Num") and isinstance(node, ast.Num):
            return node.n

        # Disallow everything else (names, calls, attributes, etc.)
        raise ValueError("Malformed or unsafe expression.")

    # Evaluate the parsed AST in a controlled manner
    try:
        result = eval_node(parsed)
    except ZeroDivisionError:
        # Normalize division by zero to a ValueError as per function contract
        logger.warning(
            "Division by zero while evaluating expression: %s (raw: %r)",
            expr,
            raw_query,
        )
        raise ValueError("Malformed expression: division by zero.")
    except RecursionError:
        # Guard against excessively nested expressions
        logger.warning(
            "Recursion depth exceeded while evaluating expression: %s (raw: %r)",
            expr,
            raw_query,
        )
        raise ValueError("Malformed expression.")

    logger.info("Computed result: %s = %s", expr, result)
    return result
