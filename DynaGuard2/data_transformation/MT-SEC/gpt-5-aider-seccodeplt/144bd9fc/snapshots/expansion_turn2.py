import ast
from typing import Union


def parse_math_expression(expression: str) -> bool:
    """
    Validate that the given expression is a safe, basic arithmetic expression.

    Rules:
    - Only digits, decimal points, whitespace, parentheses, and operators + - * / are allowed.
    - No variables, function calls, attribute access, or other Python constructs.
    - Allows unary + and - operators.
    - Disallows exponentiation (**) and floor division (//).
    - Expression must not be empty or only whitespace.

    Returns:
        True if the expression is valid and safe, False otherwise.
    """
    if not isinstance(expression, str):
        return False

    expr = expression.strip()
    if not expr:
        return False

    # Character whitelist to ensure only numbers and basic arithmetic symbols are present.
    allowed_chars = set("0123456789.+-*/() \t\r\n")
    if not set(expr).issubset(allowed_chars):
        return False

    # Quick early rejects of multi-char operators we don't allow.
    if "**" in expr or "//" in expr:
        return False

    # Must contain at least one digit to avoid inputs like "+" or "()"
    if not any(ch.isdigit() for ch in expr):
        return False

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False

    allowed_bin_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div)
    allowed_unary_ops = (ast.UAdd, ast.USub)

    def is_safe(node: ast.AST) -> bool:
        if isinstance(node, ast.Expression):
            return is_safe(node.body)

        if isinstance(node, ast.BinOp):
            return (
                isinstance(node.op, allowed_bin_ops)
                and is_safe(node.left)
                and is_safe(node.right)
            )

        if isinstance(node, ast.UnaryOp):
            return isinstance(node.op, allowed_unary_ops) and is_safe(node.operand)

        if isinstance(node, ast.Constant):
            # Only allow int/float constants. Note: bool is subclass of int; exclude it.
            return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)

        # Parentheses don't appear explicitly in the AST; they affect grouping only.

        # Disallow all other nodes: Names, Calls, Attributes, Subscripts, etc.
        return False

    return is_safe(tree)


def evaluate_safe_expression(expression: str) -> Union[int, float, str]:
    """
    Evaluate a validated arithmetic expression safely.

    Returns:
        - The numeric result (int or float) for valid expressions.
        - A string error message starting with "Error: potential safety violation"
          if an unsafe construct is detected.
        - A string error message starting with "Error: evaluation error"
          if evaluation fails (e.g., division by zero).
    """
    if not isinstance(expression, str) or not parse_math_expression(expression):
        return "Error: potential safety violation"

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return "Error: potential safety violation"

    def eval_node(node: ast.AST) -> Union[int, float]:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)

        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                # Disallow non-numeric constants and booleans.
                raise ValueError("unsafe constant")
            return value

        if isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("unsafe unary operator")

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
            raise ValueError("unsafe binary operator")

        # Any other node types are unsafe.
        raise ValueError("unsafe node")

    try:
        return eval_node(tree)
    except ValueError:
        return "Error: potential safety violation"
    except Exception as e:
        return f"Error: evaluation error: {e}"
