import re
import operator
import ast
import logging

__all__ = ["execute_calculation_string"]

logger = logging.getLogger(__name__)

# Allowed operations mapping for safe evaluation
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_VALID_NAME_RE = re.compile(r"^[A-Za-z_]\w*\Z")


def parse_expression(calc_expr: str) -> ast.AST:
    """
    Parse a calculation expression string into an AST in a safe manner.

    Args:
        calc_expr (str): Expression string to parse.

    Returns:
        ast.AST: The parsed expression tree.

    Raises:
        ValueError: If the expression is not a string or has invalid syntax.
    """
    if not isinstance(calc_expr, str):
        raise ValueError("Expression must be a string.")
    try:
        return ast.parse(calc_expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}") from None


def compute_expression(tree: ast.AST, var_dict: dict) -> float | int:
    """
    Compute the value of a parsed AST expression with variable substitution.

    Args:
        tree (ast.AST): Parsed expression tree.
        var_dict (dict): Mapping of variable names to numeric values.

    Returns:
        float | int: The numeric result.

    Raises:
        ValueError: If evaluation fails, unknown variables are referenced, or
                    unsupported nodes/operators are encountered.
    """

    def _eval(node):
        # Top-level expression
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        # Numeric literal
        if isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError("Only numeric literals are allowed.")
            return val

        # For compatibility with older Python ASTs
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
            val = node.n  # type: ignore[attr-defined]
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError("Only numeric literals are allowed.")
            return val

        # Variable name
        if isinstance(node, ast.Name):
            name = node.id
            if not _VALID_NAME_RE.match(name):
                raise ValueError(f"Invalid variable name: {name!r}")
            if name not in var_dict:
                raise ValueError(f"Unknown variable: {name!r}")
            val = var_dict[name]
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"Variable {name!r} must be a number (int or float).")
            return val

        # Unary operations: +x, -x
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_UNARYOPS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            operand = _eval(node.operand)
            try:
                result = _ALLOWED_UNARYOPS[op_type](operand)
            except Exception as e:
                raise ValueError(f"Error evaluating unary operation: {e}") from None
            _ensure_numeric(result)
            return result

        # Binary operations: +, -, *, /, //, %, **
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_BINOPS:
                raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
            left = _eval(node.left)
            right = _eval(node.right)
            try:
                result = _ALLOWED_BINOPS[op_type](left, right)
            except Exception as e:
                raise ValueError(f"Error evaluating binary operation: {e}") from None
            _ensure_numeric(result)
            return result

        # Parentheses are represented implicitly in AST by grouping; no separate node type

        # Disallow all other constructs (calls, attributes, comprehensions, etc.)
        raise ValueError(f"Unsupported expression component: {type(node).__name__}")

    try:
        result = _eval(tree)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression: {e}") from None

    _ensure_numeric(result)
    return result


def _ensure_numeric(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
        raise ValueError("Computation produced a non-numeric result.")
    if isinstance(value, complex):
        # Disallow complex results
        raise ValueError("Complex numbers are not supported.")

    # Check for infinities/NaN
    if isinstance(value, float):
        # NaN check
        if value != value:
            raise ValueError("Computation resulted in NaN.")
        # Infinity checks
        if value == float("inf") or value == float("-inf"):
            raise ValueError("Computation resulted in an infinite value.")


def _format_output_number(value: float | int) -> str:
    """Format numeric output consistently for logs and return values."""
    if isinstance(value, float) and value == 0.0:
        value = 0.0  # normalize -0.0 to 0.0
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return format(value, ".15g")
    raise ValueError("Computation did not produce a numeric result.")


def _extract_variable_names(tree: ast.AST) -> set[str]:
    """Extract variable names (loads) from the AST."""
    class _NameCollector(ast.NodeVisitor):
        def __init__(self):
            self.names: set[str] = set()

        def visit_Name(self, node: ast.Name):
            if isinstance(node.ctx, ast.Load):
                self.names.add(node.id)
            # Continue traversal
            self.generic_visit(node)

    collector = _NameCollector()
    collector.visit(tree)
    return collector.names


def execute_calculation_string(calc_expr: str, var_dict: dict) -> str:
    """
    Evaluate a mathematical expression string with variable substitution.

    Args:
        calc_expr (str): A string representing a mathematical expression.
        var_dict (dict): A dictionary mapping variable names (str) to numbers (int/float).

    Returns:
        str: The result of the simplified, computed expression as a string.

    Raises:
        ValueError: If the input expression is invalid, contains unsupported constructs,
                    references unknown variables, or the computation fails.
    """
    logger.info("Expression: %s", calc_expr)

    if var_dict is None:
        var_dict = {}
    if not isinstance(var_dict, dict):
        raise ValueError("var_dict must be a dictionary.")

    # Validate variable dictionary contents
    for k, v in var_dict.items():
        if not isinstance(k, str) or not _VALID_NAME_RE.match(k):
            raise ValueError(f"Invalid variable name: {k!r}")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"Variable {k!r} must be a number (int or float).")

    tree = parse_expression(calc_expr)

    # Prepare concise substitutions log for variables actually used in the expression
    used_vars = sorted(_extract_variable_names(tree))
    if used_vars:
        parts = []
        for name in used_vars:
            if name in var_dict:
                parts.append(f"{name}={_format_output_number(var_dict[name])}")
            else:
                parts.append(f"{name}=<MISSING>")
        logger.info("Substitutions: %s", ", ".join(parts))
    else:
        logger.info("Substitutions: (no variables)")

    result = compute_expression(tree, var_dict)

    result_str = _format_output_number(result)
    logger.info("Result: %s", result_str)
    return result_str
