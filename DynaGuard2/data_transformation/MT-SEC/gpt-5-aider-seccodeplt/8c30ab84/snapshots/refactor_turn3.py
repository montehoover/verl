import ast
import logging
import operator

# Module-level logger. Library-style: do not configure handlers globally.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Operator maps to simplify branching
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_ALLOWED_BINOPS = set(_BIN_OPS.keys()) | {ast.Pow}
_ALLOWED_UNARYOPS = set(_UNARY_OPS.keys())


def _normalize_input(expression_input: str) -> str:
    """
    Normalize and validate the raw input.
    - Must be a string
    - Must not be empty after trimming
    """
    if not isinstance(expression_input, str):
        raise ValueError("Expression must be a string.")
    expr = expression_input.strip()
    if not expr:
        raise ValueError("Expression is empty.")
    return expr


def _parse_expression(expr: str) -> ast.AST:
    """
    Parse the expression string into an AST in eval mode.
    """
    try:
        return ast.parse(expr, mode="eval")
    except Exception as e:
        raise ValueError("Invalid expression.") from e


def _is_number(value) -> bool:
    """
    Check if the value is a number (int or float) but not a bool.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_ast(node: ast.AST) -> None:
    """
    Validate the AST to ensure only safe and allowed constructs are present.
    Allowed:
      - Expression root
      - Numeric constants (int, float) excluding bool
      - Unary + and -
      - Binary operators: +, -, *, /, //, %, **
    Disallowed:
      - Names, calls, attributes, subscripts, comprehensions, lambdas, etc.
      - Bitwise ops, shifts, matrix multiplication, boolean ops, comparisons
      - Unary invert (~) or not
    """
    if isinstance(node, ast.Expression):
        _validate_ast(node.body)
        return

    if isinstance(node, ast.Constant):
        if _is_number(node.value):
            return
        raise ValueError("Illegal literal in expression.")

    # Py<3.8 compatibility: ast.Num
    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
        if _is_number(node.n):
            return
        raise ValueError("Illegal numeric literal in expression.")

    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _ALLOWED_UNARYOPS:
            raise ValueError("Illegal unary operation.")
        _validate_ast(node.operand)
        return

    if isinstance(node, ast.BinOp):
        if type(node.op) not in _ALLOWED_BINOPS:
            raise ValueError("Illegal binary operation.")
        _validate_ast(node.left)
        _validate_ast(node.right)
        return

    raise ValueError("Illegal syntax or operation in expression.")


def _eval_ast(node: ast.AST):
    """
    Evaluate a validated AST safely.
    Assumes _validate_ast has already approved the structure.
    """
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)

    if isinstance(node, ast.Constant):
        val = node.value
        if _is_number(val):
            return val
        raise ValueError("Illegal literal in expression.")

    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
        val = node.n
        if _is_number(val):
            return val
        raise ValueError("Illegal numeric literal in expression.")

    if isinstance(node, ast.UnaryOp):
        func = _UNARY_OPS.get(type(node.op))
        if func is None:
            raise ValueError("Illegal unary operation.")
        operand = _eval_ast(node.operand)
        if not _is_number(operand):
            raise ValueError("Illegal operand for unary operation.")
        return func(operand)

    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if not (_is_number(left) and _is_number(right)):
            raise ValueError("Illegal operands for binary operation.")

        # Special-case exponentiation to enforce safety limits
        if isinstance(node.op, ast.Pow):
            if abs(right) > 10:
                raise ValueError("Exponent too large.")
            if abs(left) > 1_000_000:
                raise ValueError("Base too large for exponentiation.")
            return left ** right

        func = _BIN_OPS.get(type(node.op))
        if func is None:
            raise ValueError("Illegal binary operation.")
        return func(left, right)

    raise ValueError("Illegal syntax or operation in expression.")


def compute_expression(expression_input: str):
    """
    Safely compute a simple mathematical expression provided as a string.

    Args:
        expression_input (str): The mathematical expression to evaluate.

    Returns:
        The numerical result of the evaluation (int or float).

    Raises:
        ValueError: If the expression contains illegal/unsafe constructs or cannot be safely evaluated.
    """
    logger.info("compute_expression: received input=%r", expression_input)

    try:
        # Pipeline
        expr = _normalize_input(expression_input)
        parsed = _parse_expression(expr)
        _validate_ast(parsed)
        result = _eval_ast(parsed)
    except (ZeroDivisionError, OverflowError, ValueError) as e:
        logger.warning("compute_expression: failed for input=%r, reason=%s", expression_input, str(e))
        raise ValueError("Expression cannot be safely evaluated.") from e
    except Exception as e:
        logger.exception("compute_expression: unexpected error for input=%r", expression_input)
        raise ValueError("Expression cannot be safely evaluated.") from e

    logger.info("compute_expression: result for input=%r is %r", expression_input, result)
    return result
