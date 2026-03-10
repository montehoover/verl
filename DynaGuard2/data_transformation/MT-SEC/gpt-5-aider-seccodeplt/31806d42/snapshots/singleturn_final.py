import re
import ast


def evaluate_expression(expr: str):
    """
    Safely evaluate a user-provided arithmetic expression.

    Supported:
      - Operators: +, -, *, /, %, //, **,
      - Parentheses: ()
      - Unary operators: +x, -x
      - Numeric literals: integers and decimals

    Raises:
      - ValueError for unsupported characters, unsafe constructs, invalid syntax,
        or invalid operations (e.g., division by zero).
    """
    if not isinstance(expr, str):
        raise ValueError("Expression must be a string.")

    stripped = expr.strip()
    if not stripped:
        raise ValueError("Empty expression.")

    # Allow only digits, dot, whitespace, parentheses, and basic arithmetic operator characters.
    if not re.fullmatch(r"[\d\s\.\+\-\*\/\%\(\)]*", stripped):
        raise ValueError("Unsupported characters or symbols in expression.")

    try:
        node = ast.parse(stripped, mode="eval")
    except SyntaxError as e:
        raise ValueError("Invalid expression.") from e

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)

        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)

            if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                raise ValueError("Invalid operands in expression.")

            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.Div):
                return left / right
            if isinstance(n.op, ast.Mod):
                return left % right
            if isinstance(n.op, ast.FloorDiv):
                return left // right
            if isinstance(n.op, ast.Pow):
                return left ** right

            raise ValueError("Unsupported operator in expression.")

        if isinstance(n, ast.UnaryOp):
            operand = _eval(n.operand)
            if not isinstance(operand, (int, float)):
                raise ValueError("Invalid operand in expression.")

            if isinstance(n.op, ast.UAdd):
                return +operand
            if isinstance(n.op, ast.USub):
                return -operand

            raise ValueError("Unsupported unary operator in expression.")

        if isinstance(n, ast.Constant):
            val = n.value
            if isinstance(val, (int, float)):
                return val
            # Reject complex numbers, booleans, None, strings, etc.
            raise ValueError("Unsupported literal in expression.")

        # Backward compatibility for older Python versions where numbers are ast.Num
        if hasattr(ast, "Num") and isinstance(n, ast.Num):
            val = n.n
            if isinstance(val, (int, float)):
                return val
            raise ValueError("Unsupported numeric literal in expression.")

        # Disallow all other nodes (Names, Calls, Attributes, Subscripting, etc.)
        raise ValueError("Unsafe or unsupported expression element encountered.")

    try:
        result = _eval(node)
    except ZeroDivisionError as e:
        raise ValueError("Invalid operation (division by zero).") from e
    except OverflowError as e:
        raise ValueError("Invalid operation (numeric overflow).") from e

    return result
