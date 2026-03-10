import ast
from typing import Union


def validate_expression(expression: str) -> bool:
    """
    Validate that the given expression string is a safe, basic arithmetic expression.

    Allowed:
      - Numbers (integers and floats)
      - Whitespace
      - Parentheses: ( )
      - Operators: + - * /
      - Unary operators: + and -

    Disallowed:
      - Any names/variables (e.g., x, y, pi)
      - Function calls (e.g., sin(1), __import__('os'))
      - Attribute access (e.g., obj.attr)
      - Subscripts (e.g., a[0])
      - Comparisons, boolean ops, bitwise ops
      - Modulo %, floor division //, exponentiation **

    Returns:
        True if the expression is valid and safe, False otherwise.
    """
    if not isinstance(expression, str):
        return False

    try:
        # Parse strictly as an expression, not a statement
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return False

    def is_number_node(node: ast.AST) -> bool:
        # Py3.8+: numbers are ast.Constant with int/float; exclude bool
        if isinstance(node, ast.Constant):
            return (isinstance(node.value, (int, float)) and not isinstance(node.value, bool))
        # Older versions: ast.Num
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
            return (isinstance(node.n, (int, float)) and not isinstance(node.n, bool))  # type: ignore[attr-defined]
        return False

    allowed_bin_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div)
    allowed_unary_ops = (ast.UAdd, ast.USub)

    def is_safe(node: ast.AST) -> bool:
        # Root expression
        if isinstance(node, ast.Expression):
            return is_safe(node.body)

        # Binary operations: left <op> right
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, allowed_bin_ops):
                return False
            return is_safe(node.left) and is_safe(node.right)

        # Unary operations: +x, -x
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, allowed_unary_ops):
                return False
            return is_safe(node.operand)

        # Numeric literals
        if is_number_node(node):
            return True

        # Parentheses do not create a special node; they affect tree shape only.
        # Any other node types are unsafe.
        return False

    return is_safe(tree)
