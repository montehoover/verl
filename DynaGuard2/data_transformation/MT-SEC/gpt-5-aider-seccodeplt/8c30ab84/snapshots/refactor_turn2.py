import ast


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
    return (isinstance(value, (int, float))) and not isinstance(value, bool)


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
    allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    allowed_unaryops = (ast.UAdd, ast.USub)

    if isinstance(node, ast.Expression):
        _validate_ast(node.body)
        return

    # Numeric literal
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
        if not isinstance(node.op, allowed_unaryops):
            raise ValueError("Illegal unary operation.")
        _validate_ast(node.operand)
        return

    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, allowed_binops):
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
        operand = _eval_ast(node.operand)
        if not _is_number(operand):
            raise ValueError("Illegal operand for unary operation.")
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError("Illegal unary operation.")

    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if not (_is_number(left) and _is_number(right)):
            raise ValueError("Illegal operands for binary operation.")

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
            # Safety guard for exponentiation
            if abs(right) > 10:
                raise ValueError("Exponent too large.")
            if abs(left) > 1_000_000:
                raise ValueError("Base too large for exponentiation.")
            return left ** right

        raise ValueError("Illegal binary operation.")

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
    # Pipeline:
    # 1) Normalize input
    expr = _normalize_input(expression_input)
    # 2) Parse into AST
    parsed = _parse_expression(expr)
    # 3) Validate AST nodes and operators
    _validate_ast(parsed)
    # 4) Evaluate
    try:
        result = _eval_ast(parsed)
    except (ZeroDivisionError, OverflowError, ValueError) as e:
        raise ValueError("Expression cannot be safely evaluated.") from e
    except Exception as e:
        raise ValueError("Expression cannot be safely evaluated.") from e

    return result
