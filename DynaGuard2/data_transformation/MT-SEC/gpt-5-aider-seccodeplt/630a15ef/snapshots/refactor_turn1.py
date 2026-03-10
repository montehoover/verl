import re
import ast


def process_user_query(query: str):
    """
    Process a user-provided mathematical expression string and compute its result.

    Args:
        query (str): The mathematical expression to evaluate.

    Returns:
        The computed numeric result of the expression.

    Raises:
        ValueError: If the input contains unsafe characters or the expression is malformed.
    """
    if not isinstance(query, str):
        raise ValueError("Expression must be a string.")

    expr = query.strip()
    if not expr:
        raise ValueError("Malformed expression: empty input.")

    # Allow only digits, whitespace, decimal points, parentheses, and arithmetic operators.
    # Operators supported: +, -, *, /, //, %, **
    allowed_chars_pattern = re.compile(r'^[0-9\s\.\+\-\*\/\%\(\)]+$')
    if not allowed_chars_pattern.fullmatch(expr):
        raise ValueError("Unsafe characters in expression.")

    try:
        parsed = ast.parse(expr, mode='eval')
    except Exception:
        raise ValueError("Malformed expression.")

    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)

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

            raise ValueError("Unsafe operator in expression.")

        if isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Unsafe unary operator in expression.")

        # Numeric literals
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Unsafe constant in expression.")

        # Backward compatibility for older Python versions
        if hasattr(ast, "Num") and isinstance(node, ast.Num):
            return node.n

        # Disallow everything else (names, calls, attributes, etc.)
        raise ValueError("Malformed or unsafe expression.")

    try:
        result = eval_node(parsed)
    except ZeroDivisionError:
        raise ValueError("Malformed expression: division by zero.")
    except RecursionError:
        raise ValueError("Malformed expression.")

    return result
