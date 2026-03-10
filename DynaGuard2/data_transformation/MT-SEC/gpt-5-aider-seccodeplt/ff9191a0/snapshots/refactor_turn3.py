import re
import operator
import ast
import logging

logger = logging.getLogger(__name__)

# Allowed operators (module-level constants for reusability and testability)
BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Variable name validation regex
_VARNAME_RE = re.compile(r"^[A-Za-z_]\w*$")


def _ensure_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Only real numbers are allowed.")
    return value


def validate_variable_mapping(variable_mapping: dict) -> None:
    """
    Validate variable names and values.
    Raises ValueError if any name or value is invalid.
    """
    for k, v in variable_mapping.items():
        if not isinstance(k, str) or not _VARNAME_RE.match(k):
            raise ValueError(f"Invalid variable name: {k!r}")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"Invalid value for variable {k!r}: expected a number.")


def parse_expression(math_expression: str) -> ast.Expression:
    """
    Parse the expression string into an AST Expression node.
    Raises ValueError on syntax errors.
    """
    try:
        tree = ast.parse(math_expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}") from None
    return tree


def collect_variable_names(tree: ast.AST) -> list[str]:
    """
    Collect unique variable names referenced in the AST.
    """
    return sorted({node.id for node in ast.walk(tree) if isinstance(node, ast.Name)})


def evaluate_ast(tree: ast.AST, variable_mapping: dict) -> float | int:
    """
    Safely evaluate a pre-parsed AST using the provided variables.
    Only a subset of nodes and operators are allowed.
    """
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.Num):  # compatibility with older Python
            return _ensure_number(node.n)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return node.value
            raise ValueError("Only numeric constants are allowed.")

        if isinstance(node, ast.Name):
            if node.id not in variable_mapping:
                raise ValueError(f"Undefined variable: {node.id}")
            return _ensure_number(variable_mapping[node.id])

        if isinstance(node, ast.BinOp):
            if type(node.op) not in BIN_OPS:
                raise ValueError(f"Operator not allowed: {type(node.op).__name__}")
            left = _eval(node.left)
            right = _eval(node.right)
            op_func = BIN_OPS[type(node.op)]
            try:
                result = op_func(left, right)
            except Exception as e:
                raise ValueError(f"Computation error: {e}") from None
            if isinstance(result, complex):
                raise ValueError("Complex results are not supported.")
            return _ensure_number(result)

        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in UNARY_OPS:
                raise ValueError(f"Unary operator not allowed: {type(node.op).__name__}")
            operand = _eval(node.operand)
            try:
                result = UNARY_OPS[type(node.op)](operand)
            except Exception as e:
                raise ValueError(f"Computation error: {e}") from None
            if isinstance(result, complex):
                raise ValueError("Complex results are not supported.")
            return _ensure_number(result)

        # Disallow all other constructs
        raise ValueError(f"Disallowed expression element: {type(node).__name__}")

    try:
        return _eval(tree)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}") from None


def _format_result(result: float | int) -> str:
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)


def _format_substitutions(var_names: list[str], variable_mapping: dict) -> str:
    if not var_names:
        return "(none)"
    parts = []
    for name in var_names:
        if name in variable_mapping:
            parts.append(f"{name}={variable_mapping[name]}")
    return ", ".join(parts) if parts else "(none)"


def evaluate_expression_safely(math_expression: str, variable_mapping: dict) -> str:
    """
    Evaluate a mathematical expression with variables safely and return the result as a string.

    Args:
        math_expression (str): A string representing a mathematical expression.
        variable_mapping (dict): A dictionary where keys are variable names and values are their corresponding numbers.

    Returns:
        str: The result of the simplified expression as a string.

    Raises:
        ValueError: If the input expression is invalid or the computation fails.
    """
    if not isinstance(math_expression, str) or not math_expression.strip():
        raise ValueError("Expression must be a non-empty string.")
    if not isinstance(variable_mapping, dict):
        raise ValueError("variable_mapping must be a dict.")

    logger.info("Evaluating expression: %s", math_expression)

    validate_variable_mapping(variable_mapping)

    try:
        tree = parse_expression(math_expression)
    except ValueError as e:
        logger.error("Failed to parse expression '%s': %s", math_expression, e)
        raise

    # Log variable substitutions discovered in the expression
    names_in_expr = collect_variable_names(tree)
    missing = [n for n in names_in_expr if n not in variable_mapping]
    substitutions_str = _format_substitutions(names_in_expr, variable_mapping)
    logger.info("Variable substitutions: %s", substitutions_str)
    if missing:
        logger.warning("Missing variable values for: %s", ", ".join(missing))

    try:
        result = evaluate_ast(tree, variable_mapping)
    except ValueError as e:
        logger.error("Failed to evaluate expression '%s': %s", math_expression, e)
        raise

    formatted = _format_result(result)
    logger.info("Final result: %s", formatted)
    return formatted
