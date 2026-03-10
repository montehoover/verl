import ast
import operator

def evaluate_math_expression(expression):
    """
    Safely evaluate a simple expression string supporting:
      - Arithmetic: +, -, *, / with integers/floats
      - Strings: concatenation with + and repetition with * (e.g., 'a' * 3 or 3 * 'a')
      - Parentheses
    Returns the result (number or string), or 'Error!' if the expression is invalid or unsafe.
    """
    if not isinstance(expression, str):
        return 'Error!'
    expr = expression.strip()
    if not expr:
        return 'Error!'

    def is_number(val):
        return type(val) in (int, float)

    def is_string(val):
        return isinstance(val, str)

    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)

        # Numbers (ints/floats) and strings only
        if isinstance(node, ast.Constant):
            if is_number(node.value) or is_string(node.value):
                return node.value
            raise ValueError("Unsupported constant type")
        # Backward compatibility with older Python versions
        if isinstance(node, ast.Num):  # type: ignore[attr-defined]
            return node.n  # type: ignore[union-attr]
        if isinstance(node, ast.Str):  # type: ignore[attr-defined]
            return node.s  # type: ignore[union-attr]

        # Unary operations (+x, -x) for numbers only
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            val = eval_node(node.operand)
            if not is_number(val):
                raise ValueError("Unary operators only allowed on numbers")
            return +val if isinstance(node.op, ast.UAdd) else -val

        # Binary operations
        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)

            # Addition: number+number or str+str
            if isinstance(node.op, ast.Add):
                if is_number(left) and is_number(right):
                    return left + right
                if is_string(left) and is_string(right):
                    return left + right
                raise ValueError("Invalid types for +")

            # Subtraction: number-number
            if isinstance(node.op, ast.Sub):
                if is_number(left) and is_number(right):
                    return left - right
                raise ValueError("Invalid types for -")

            # Multiplication: number*number, str*int, int*str
            if isinstance(node.op, ast.Mult):
                if is_number(left) and is_number(right):
                    return left * right
                if (is_string(left) and isinstance(right, int)) or (isinstance(left, int) and is_string(right)):
                    return left * right
                raise ValueError("Invalid types for *")

            # Division: number/number
            if isinstance(node.op, ast.Div):
                if is_number(left) and is_number(right):
                    if right == 0:
                        raise ZeroDivisionError()
                    return operator.truediv(left, right)
                raise ValueError("Invalid types for /")

            # Any other operator is disallowed
            raise ValueError("Disallowed operator")

        # Explicitly reject all other nodes (Names, Calls, Attributes, etc.)
        raise ValueError("Disallowed expression")

    try:
        parsed = ast.parse(expr, mode='eval')
        result = eval_node(parsed)
        return result
    except Exception:
        return 'Error!'
