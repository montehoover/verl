import re
import ast
import logging


# Configure a file-based logger in the current working directory.
logger = logging.getLogger("expression_evaluator")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _file_handler = logging.FileHandler("expression_evaluator.log", mode="a", encoding="utf-8")
    _file_handler.setLevel(logging.INFO)
    _formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    _file_handler.setFormatter(_formatter)
    logger.addHandler(_file_handler)
    logger.propagate = False


# Precompiled whitelist regex for performance and clarity.
# Allows digits, decimal point, whitespace, parentheses, and basic arithmetic operators.
_ALLOWED_CHAR_PATTERN = re.compile(r"[0-9\.\s\+\-\*\/\%\(\)]+")


def _normalize_expression(math_expr: str) -> str:
    """
    Normalize the input expression by validating its type and trimming whitespace.

    :param math_expr: The raw input expression; must be a string.
    :return: A trimmed expression string.
    :raises ValueError: If the input is not a string or is empty after trimming.
    """
    if not isinstance(math_expr, str):
        raise ValueError("Expression must be a string.")

    expr = math_expr.strip()

    if not expr:
        raise ValueError("Empty expression.")

    return expr


def _validate_characters(expr: str) -> None:
    """
    Validate that the expression contains only allowed characters.

    This acts as a quick pre-filter before AST parsing to reject obviously unsafe inputs.

    :param expr: Expression string to validate.
    :raises ValueError: If disallowed characters or operators are present.
    """
    if not _ALLOWED_CHAR_PATTERN.fullmatch(expr):
        raise ValueError("Expression contains unsupported characters or operators.")
    logger.debug("Character validation passed for expression: %r", expr)


def _parse_expression_to_ast(expr: str) -> ast.AST:
    """
    Parse the expression string into an AST in eval mode.

    :param expr: Validated expression string.
    :return: AST of the parsed expression.
    :raises ValueError: If parsing fails.
    """
    try:
        tree = ast.parse(expr, mode="eval")
        logger.debug("Parsed AST successfully: %s", type(tree).__name__)
        return tree
    except Exception as e:
        # Wrap any parsing error in ValueError to comply with the contract.
        logger.error("AST parsing failed for %r: %s", expr, e, exc_info=True)
        raise ValueError("Failed to parse expression.") from e


def _is_number(value) -> bool:
    """
    Check if a value is a numeric type we accept (int or float).

    :param value: Any Python value.
    :return: True if value is int or float, else False.
    """
    return isinstance(value, (int, float))


def _eval_number_node(node: ast.AST):
    """
    Evaluate a numeric literal node.

    Supports ast.Constant with int/float and (for older Python) ast.Num.

    :param node: AST node expected to represent a numeric literal.
    :return: The numeric value.
    :raises ValueError: If the node isn't a supported numeric literal.
    """
    if isinstance(node, ast.Constant):
        if _is_number(node.value):
            logger.debug("Number literal: %s", node.value)
            return node.value
        raise ValueError("Unsupported constant in expression.")

    # Backward compatibility for older Python versions where numbers are ast.Num
    if hasattr(ast, "Num") and isinstance(node, ast.Num):
        logger.debug("Number literal (ast.Num): %s", node.n)
        return node.n

    raise ValueError("Unsupported number literal.")


def _eval_unary_node(node: ast.UnaryOp):
    """
    Evaluate a unary operation node (supported: +, -).

    :param node: ast.UnaryOp node.
    :return: Result of applying the unary operator to its operand.
    :raises ValueError: If operator or operand is unsupported.
    """
    operand = _eval_node(node.operand)

    if not _is_number(operand):
        raise ValueError("Invalid operand for unary operation.")

    # Dispatch table for supported unary operators, with symbols for logging.
    unary_dispatch = {
        ast.UAdd: ("+", lambda a: +a),
        ast.USub: ("-", lambda a: -a),
    }

    for op_type, (symbol, func) in unary_dispatch.items():
        if isinstance(node.op, op_type):
            result = func(operand)
            logger.debug("UnaryOp: %s%s -> %s", symbol, operand, result)
            return result

    # Any other unary operator is disallowed.
    raise ValueError("Unsupported unary operator.")


def _eval_binary_node(node: ast.BinOp):
    """
    Evaluate a binary operation node (supported: +, -, *, /, //, %).

    :param node: ast.BinOp node.
    :return: Result of applying the operator to the left and right operands.
    :raises ValueError: If operator or operands are unsupported.
    """
    left = _eval_node(node.left)
    right = _eval_node(node.right)

    if not (_is_number(left) and _is_number(right)):
        raise ValueError("Invalid operands for binary operation.")

    # Dispatch table for supported binary operators, with symbols for logging.
    bin_dispatch = {
        ast.Add: ("+", lambda a, b: a + b),
        ast.Sub: ("-", lambda a, b: a - b),
        ast.Mult: ("*", lambda a, b: a * b),
        ast.Div: ("/", lambda a, b: a / b),
        ast.FloorDiv: ("//", lambda a, b: a // b),
        ast.Mod: ("%", lambda a, b: a % b),
    }

    for op_type, (symbol, func) in bin_dispatch.items():
        if isinstance(node.op, op_type):
            result = func(left, right)
            logger.debug("BinOp: %s %s %s -> %s", left, symbol, right, result)
            return result

    # Explicitly disallow power (**) and any other operator not listed.
    raise ValueError("Unsupported operator in expression.")


def _eval_node(node: ast.AST):
    """
    Recursively evaluate supported AST nodes.

    Supported nodes:
      - ast.Expression (root wrapper)
      - ast.Constant / ast.Num (numeric literals)
      - ast.UnaryOp (UAdd, USub)
      - ast.BinOp (Add, Sub, Mult, Div, FloorDiv, Mod)

    :param node: AST node to evaluate.
    :return: The computed numeric value.
    :raises ValueError: If node type is unsupported or invalid.
    """
    # Root expression container
    if isinstance(node, ast.Expression):
        logger.debug("Evaluating ast.Expression node")
        return _eval_node(node.body)

    # Numeric literals
    if isinstance(node, (ast.Constant, getattr(ast, "Num", ()))):
        return _eval_number_node(node)

    # Unary operations: +, -
    if isinstance(node, ast.UnaryOp):
        return _eval_unary_node(node)

    # Binary operations: +, -, *, /, //, %
    if isinstance(node, ast.BinOp):
        return _eval_binary_node(node)

    # Any other node types (e.g., Name, Call, Attribute, etc.) are disallowed
    raise ValueError("Unsupported expression content.")


def _evaluate_ast(tree: ast.AST):
    """
    Safely evaluate a previously parsed AST.

    :param tree: AST generated from parsing a validated expression.
    :return: Numeric result of the evaluation.
    :raises ValueError: If evaluation fails or expression contains unsupported constructs.
    """
    try:
        logger.debug("Starting AST evaluation.")
        result = _eval_node(tree)
        logger.debug("AST evaluation completed with result: %s", result)
        return result
    except ValueError:
        # Re-raise known validation errors unchanged.
        raise
    except Exception as e:
        # Catch-all to prevent leaking internal exceptions.
        raise ValueError("Evaluation failed.") from e


def evaluate_expression(math_expr: str):
    """
    Safely evaluate a mathematical expression string consisting of basic arithmetic
    operators and parentheses.

    Supported:
      - Binary operators: +, -, *, /, //, %
      - Unary operators: +, -
      - Parentheses: ( )

    Not supported (will raise ValueError): power (**), bitwise ops, function calls,
    names/variables, attribute access, and any non-numeric/unsafe constructs.

    :param math_expr: str - a string containing a mathematical expression to evaluate
    :return: evaluated numeric result (int or float)
    :raises ValueError: on unsupported operators/characters or evaluation failure
    """
    logger.info("Evaluating expression: %r", math_expr)

    try:
        # Normalize and validate the raw input string.
        expr = _normalize_expression(math_expr)
        logger.debug("Normalized expression: %r", expr)

        # Ensure only safe characters are present before parsing.
        _validate_characters(expr)

        # Parse to AST using Python's built-in parser in expression mode.
        tree = _parse_expression_to_ast(expr)

        # Evaluate the AST with a strict whitelist of supported nodes/operators.
        result = _evaluate_ast(tree)

        logger.info("Evaluation succeeded: %r = %s", expr, result)
        return result

    except ValueError as e:
        logger.error("Evaluation failed for %r: %s", math_expr, e, exc_info=True)
        raise
    except Exception as e:
        logger.error("Unexpected error during evaluation for %r: %s", math_expr, e, exc_info=True)
        raise ValueError("Evaluation failed.") from e
