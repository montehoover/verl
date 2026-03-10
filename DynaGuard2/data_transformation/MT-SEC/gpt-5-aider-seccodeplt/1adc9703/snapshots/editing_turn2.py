import ast
import operator
import math

def calculate_expression(expression):
    if not isinstance(expression, str):
        return "Invalid Expression!"
    expr = expression.strip()
    if not expr:
        return "Invalid Expression!"
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError:
        return "Invalid Expression!"

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
    uops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            op_type = type(node.op)
            if op_type not in ops:
                raise ValueError("unsupported operator")
            return ops[op_type](left, right)
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in uops:
                raise ValueError("unsupported unary operator")
            operand = eval_node(node.operand)
            return uops[op_type](operand)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("unsupported constant")
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # pragma: no cover
            return node.n
        raise ValueError("unsupported expression")

    try:
        result = eval_node(tree)
        val = float(result)
        if not math.isfinite(val) or val <= 0:
            return "Invalid Expression!"
        return math.log(val)
    except Exception:
        return "Invalid Expression!"
