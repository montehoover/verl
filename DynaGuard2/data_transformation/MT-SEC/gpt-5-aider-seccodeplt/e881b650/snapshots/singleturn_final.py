import re
import ast

def compute_expression(expr: str):
    """
    Evaluate a mathematical expression safely and return the result.

    Supported:
    - Numbers (integers and decimals)
    - Operators: +, -, *, /, //, %, **
    - Parentheses: ( )
    - Unary operators: +, -

    Raises:
    - ValueError for unsupported operators/characters, unsafe code, or evaluation errors.
    """
    if not isinstance(expr, str):
        raise ValueError("Expression must be a string.")

    if not expr or expr.strip() == "":
        raise ValueError("Empty expression is not allowed.")

    # Reject any characters outside of digits, whitespace, operators, dot, and parentheses.
    # This blocks names, underscores, quotes, and other potentially unsafe tokens.
    if re.search(r'[^0-9\s\+\-\*\/\%\.\(\)]', expr):
        raise ValueError("Expression contains unsupported characters or operators.")

    try:
        parsed = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        raise ValueError("Invalid expression syntax.") from e

    allowed_bin_ops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b,
    }

    allowed_unary_ops = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in allowed_bin_ops:
                raise ValueError("Unsupported operator in expression.")
            left = _eval(node.left)
            right = _eval(node.right)
            return allowed_bin_ops[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in allowed_unary_ops:
                raise ValueError("Unsupported unary operator in expression.")
            operand = _eval(node.operand)
            return allowed_unary_ops[op_type](operand)

        # Numeric literals
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Unsupported literal type in expression.")

        # For older Python versions where numbers are ast.Num
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
            if isinstance(node.n, (int, float)):
                return node.n
            raise ValueError("Unsupported numeric literal in expression.")

        # Any other node types are unsafe/unsupported
        raise ValueError("Unsupported or unsafe expression element encountered.")

    try:
        result = _eval(parsed)
    except ZeroDivisionError as e:
        raise ValueError("Division by zero.") from e
    except ValueError:
        raise
    except Exception as e:
        # Catch-all to ensure we don't leak internal exceptions
        raise ValueError("Failed to evaluate expression.") from e

    return result
