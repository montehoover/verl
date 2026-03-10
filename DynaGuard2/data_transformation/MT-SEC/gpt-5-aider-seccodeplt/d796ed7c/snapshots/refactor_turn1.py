import ast
import operator


def secure_math_eval(exp_str: str):
    """
    Securely evaluate a simple mathematical expression from a user-provided string.

    Allowed:
      - Numbers (int, float, including scientific notation)
      - Binary operations: +, -, *, /, //, %
      - Unary operations: +, -
      - Parentheses for grouping

    Disallowed (will raise ValueError):
      - Power (**)
      - Bitwise operations (&, |, ^, ~, <<, >>)
      - Comparisons, boolean ops, conditionals
      - Names, calls, attributes, subscripts, collections, etc.

    Raises:
      - ValueError on invalid/unsafe input
      - ZeroDivisionError if a division by zero occurs
    """
    if not isinstance(exp_str, str):
        raise ValueError("Expression must be a string")

    # Basic size/complexity guard
    if len(exp_str) > 1000:
        raise ValueError("Expression too long")

    try:
        tree = ast.parse(exp_str, mode="eval")
    except SyntaxError:
        raise ValueError("Invalid expression")

    allowed_binops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        # ast.Pow is intentionally not allowed to avoid excessive computation / DoS
    }
    allowed_unaryops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def eval_node(node, depth=0):
        if depth > 100:
            raise ValueError("Expression too complex")

        if isinstance(node, ast.Expression):
            return eval_node(node.body, depth + 1)

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in allowed_binops:
                raise ValueError("Restricted operator")
            left = eval_node(node.left, depth + 1)
            right = eval_node(node.right, depth + 1)
            if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                raise ValueError("Invalid operand")
            return allowed_binops[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in allowed_unaryops:
                raise ValueError("Restricted operator")
            operand = eval_node(node.operand, depth + 1)
            if not isinstance(operand, (int, float)):
                raise ValueError("Invalid operand")
            return allowed_unaryops[op_type](operand)

        # Numeric constants
        if isinstance(node, ast.Constant):
            val = node.value
            # Exclude bool (bool is a subclass of int)
            if isinstance(val, bool):
                raise ValueError("Invalid literal")
            if isinstance(val, (int, float)):
                return val
            raise ValueError("Invalid literal")

        # Compatibility for older Python versions
        if hasattr(ast, "Num") and isinstance(node, ast.Num):
            val = node.n
            if isinstance(val, bool):
                raise ValueError("Invalid literal")
            if isinstance(val, (int, float)):
                return val
            raise ValueError("Invalid literal")

        # Anything else is restricted
        raise ValueError("Restricted expression")

    return eval_node(tree)
