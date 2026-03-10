import ast
from typing import Any

# Configuration constants to keep limits and policies centralized
MAX_EXPR_LENGTH = 1000
MAX_AST_NODES = 1000
MAX_RECURSION_DEPTH = 50
ALLOWED_NAMES = {"True", "False"}

# Allowlist of AST node types permitted in user expressions
ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Name,
    ast.Load,
    ast.Constant,
    # Operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.UAdd, ast.USub,
    ast.And, ast.Or,
    ast.Not,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)


def _normalize_and_check_input(input_expr: str) -> str:
    """
    Normalize and validate the raw input string.

    - Ensures the input is a string.
    - Strips surrounding whitespace.
    - Checks for non-empty content and length limits.

    Returns:
        The normalized expression string.

    Raises:
        ValueError: If the input is invalid.
    """
    if not isinstance(input_expr, str):
        raise ValueError("Input must be a string")
    expr = input_expr.strip()
    if not expr:
        raise ValueError("Empty expression")
    if len(expr) > MAX_EXPR_LENGTH:
        raise ValueError("Expression too long")
    return expr


def _parse_expression(expr: str) -> ast.AST:
    """
    Parse the expression string into a Python AST (in eval mode).

    Returns:
        An AST representing the expression.

    Raises:
        ValueError: If parsing fails.
    """
    try:
        return ast.parse(expr, mode="eval")
    except Exception as exc:
        raise ValueError("Parsing failed") from exc


def _validate_expression_ast(tree: ast.AST) -> None:
    """
    Validate the AST to ensure only safe nodes and identifiers are present.

    - Enforces an allowlist of AST node types.
    - Limits the total number of nodes to avoid pathological inputs.
    - Restricts variable names to a small safe set (True/False).

    Raises:
        ValueError: If the AST contains disallowed constructs or exceeds limits.
    """
    node_count = 0
    for node in ast.walk(tree):
        node_count += 1
        if node_count > MAX_AST_NODES:
            raise ValueError("AST too large")
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"Disallowed node: {type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id not in ALLOWED_NAMES:
                raise ValueError("Disallowed identifier")


def _evaluate_expression_ast(tree: ast.AST) -> Any:
    """
    Evaluate a previously validated AST safely.

    Supported:
    - Numeric arithmetic: +, -, *, /, //, %
    - Unary plus/minus: +x, -x
    - Boolean logic: and, or, not
    - Comparisons: ==, !=, <, <=, >, >= with proper type checks
    - Constants: ints, floats, booleans
    - Names: True, False

    Returns:
        The computed Python value.

    Raises:
        ValueError: For any evaluation errors or type violations.
    """
    def eval_node(node: ast.AST, depth: int = 0) -> Any:
        if depth > MAX_RECURSION_DEPTH:
            raise ValueError("Expression too deep")

        if isinstance(node, ast.Expression):
            return eval_node(node.body, depth + 1)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (bool, int, float)):
                return node.value
            raise ValueError("Disallowed constant")

        if isinstance(node, ast.Name):
            if node.id == "True":
                return True
            if node.id == "False":
                return False
            raise ValueError("Disallowed name")

        if isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand, depth + 1)
            if isinstance(node.op, ast.Not):
                if isinstance(operand, bool):
                    return not operand
                raise ValueError("not expects boolean")
            if isinstance(node.op, (ast.UAdd, ast.USub)):
                if isinstance(operand, (int, float)) and not isinstance(operand, bool):
                    return +operand if isinstance(node.op, ast.UAdd) else -operand
                raise ValueError("Unary op expects number")
            raise ValueError("Disallowed unary operator")

        if isinstance(node, ast.BinOp):
            left = eval_node(node.left, depth + 1)
            right = eval_node(node.right, depth + 1)

            # Disallow booleans in arithmetic and ensure numeric operands
            if (isinstance(left, bool) or isinstance(right, bool) or
                    not isinstance(left, (int, float)) or
                    not isinstance(right, (int, float))):
                raise ValueError("Arithmetic expects numbers")

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
            # Any other binary op (e.g., power, bitwise) is disallowed
            raise ValueError("Disallowed binary operator")

        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                result = True
                for value_node in node.values:
                    val = eval_node(value_node, depth + 1)
                    if not isinstance(val, bool):
                        raise ValueError("and/or expects booleans")
                    result = result and val
                return result
            if isinstance(node.op, ast.Or):
                result = False
                for value_node in node.values:
                    val = eval_node(value_node, depth + 1)
                    if not isinstance(val, bool):
                        raise ValueError("and/or expects booleans")
                    result = result or val
                return result
            raise ValueError("Disallowed boolean operator")

        if isinstance(node, ast.Compare):
            left = eval_node(node.left, depth + 1)
            overall = True
            prev = left
            for op, comp_node in zip(node.ops, node.comparators):
                right = eval_node(comp_node, depth + 1)

                # Enforce type compatibility:
                # - Compare bools with bools
                # - Compare numbers with numbers
                if isinstance(prev, bool) or isinstance(right, bool):
                    if not (isinstance(prev, bool) and isinstance(right, bool)):
                        raise ValueError("Invalid comparison types")
                else:
                    if not (isinstance(prev, (int, float)) and isinstance(right, (int, float))):
                        raise ValueError("Invalid comparison types")

                if isinstance(op, ast.Eq):
                    ok = prev == right
                elif isinstance(op, ast.NotEq):
                    ok = prev != right
                elif isinstance(op, ast.Lt):
                    ok = prev < right
                elif isinstance(op, ast.LtE):
                    ok = prev <= right
                elif isinstance(op, ast.Gt):
                    ok = prev > right
                elif isinstance(op, ast.GtE):
                    ok = prev >= right
                else:
                    raise ValueError("Disallowed comparison operator")

                if not ok:
                    overall = False
                prev = right
            return overall

        # Any other node type should not occur after validation
        raise ValueError(f"Disallowed syntax: {type(node).__name__}")

    return eval_node(tree)


def calculate_user_input(input_expr: str) -> str:
    """
    Safely evaluate a user-supplied arithmetic or logical expression.

    Steps:
    1) Normalize and validate the input string.
    2) Parse into an AST (eval mode).
    3) Validate the AST against a strict allowlist.
    4) Evaluate the AST using a safe interpreter.
    5) Return the result as a string; on any failure, return 'Processing Failed!'.

    Args:
        input_expr: A string representing the expression to evaluate.

    Returns:
        str: The result of the evaluated expression as a string, or
             'Processing Failed!' if evaluation fails or the input is unsafe.
    """
    try:
        expr = _normalize_and_check_input(input_expr)
        tree = _parse_expression(expr)
        _validate_expression_ast(tree)
        result = _evaluate_expression_ast(tree)
        return str(result)
    except Exception:
        return 'Processing Failed!'
