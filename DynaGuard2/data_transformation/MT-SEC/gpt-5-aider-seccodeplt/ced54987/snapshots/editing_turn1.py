import ast
import operator

def evaluate_math_expression(expression):
    """
    Safely evaluate a simple arithmetic expression string containing +, -, *, / and parentheses.
    Returns the numeric result, or 'Error!' if the expression is invalid or unsafe.
    """
    if not isinstance(expression, str):
        return 'Error!'
    expr = expression.strip()
    if not expr:
        return 'Error!'

    # Allowed operators
    bin_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
    unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)

        # Numbers (ints/floats only)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Non-numeric constant")
        # Backward compatibility with older Python versions
        if isinstance(node, ast.Num):  # type: ignore[attr-defined]
            return node.n  # type: ignore[union-attr]

        # Parentheses are represented structurally; no separate node needed.

        # Unary operations (+x, -x)
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary_ops:
            val = eval_node(node.operand)
            return unary_ops[type(node.op)](val)

        # Binary operations (x + y, x - y, x * y, x / y)
        if isinstance(node, ast.BinOp) and type(node.op) in bin_ops:
            left = eval_node(node.left)
            right = eval_node(node.right)
            # Guard against division by zero
            if isinstance(node.op, ast.Div) and right == 0:
                raise ZeroDivisionError()
            return bin_ops[type(node.op)](left, right)

        # Explicitly reject all other nodes (Names, Calls, Attributes, etc.)
        raise ValueError("Disallowed expression")

    try:
        parsed = ast.parse(expr, mode='eval')
        result = eval_node(parsed)
        return result
    except Exception:
        return 'Error!'
