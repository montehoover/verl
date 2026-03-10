import ast

def evaluate_expression_safely(input_expr: str) -> str:
    """
    Safely evaluate arithmetic or logic expressions provided in input_expr.
    Allowed:
      - Numbers (ints, floats)
      - Booleans: True, False
      - Arithmetic: +, -, *, /, //, %, **, unary +/-
      - Comparisons: ==, !=, <, <=, >, >= (chained comparisons allowed)
      - Boolean operations: and, or, not
      - Parentheses

    Any invalid, unsafe input or runtime error returns "Processing Failed!".
    """
    try:
        if not isinstance(input_expr, str):
            return "Processing Failed!"

        s = input_expr.strip()
        if not s:
            return "Processing Failed!"

        node = ast.parse(s, mode="eval")

        def eval_node(n):
            if isinstance(n, ast.Expression):
                return eval_node(n.body)

            # Constants
            if isinstance(n, ast.Constant):
                v = n.value
                if isinstance(v, (int, float, bool)):
                    return v
                raise ValueError("Invalid constant type")

            # Backward compatibility for older AST nodes
            if isinstance(n, ast.Num):  # type: ignore[attr-defined]
                return n.n
            if isinstance(n, ast.NameConstant):  # type: ignore[attr-defined]
                if n.value in (True, False):
                    return n.value
                raise ValueError("Invalid name constant")

            # Names allowed: True/False only
            if isinstance(n, ast.Name):
                if n.id == "True":
                    return True
                if n.id == "False":
                    return False
                raise ValueError("Invalid identifier")

            # Unary operations
            if isinstance(n, ast.UnaryOp):
                operand = eval_node(n.operand)
                if isinstance(n.op, ast.Not):
                    if isinstance(operand, bool):
                        return not operand
                    raise ValueError("not requires boolean")
                if isinstance(n.op, (ast.UAdd, ast.USub)):
                    if isinstance(operand, (int, float)) and not isinstance(operand, bool):
                        return +operand if isinstance(n.op, ast.UAdd) else -operand
                    raise ValueError("Unary +/- requires number")
                raise ValueError("Unsupported unary operator")

            # Binary arithmetic operations
            if isinstance(n, ast.BinOp):
                left = eval_node(n.left)
                right = eval_node(n.right)
                if not (isinstance(left, (int, float)) and isinstance(right, (int, float))):
                    raise ValueError("Arithmetic operands must be numbers")

                if isinstance(n.op, ast.Add):
                    return left + right
                if isinstance(n.op, ast.Sub):
                    return left - right
                if isinstance(n.op, ast.Mult):
                    return left * right
                if isinstance(n.op, ast.Div):
                    return left / right
                if isinstance(n.op, ast.FloorDiv):
                    return left // right
                if isinstance(n.op, ast.Mod):
                    return left % right
                if isinstance(n.op, ast.Pow):
                    return left ** right
                raise ValueError("Unsupported binary operator")

            # Boolean operations
            if isinstance(n, ast.BoolOp):
                values = [eval_node(v) for v in n.values]
                if not all(isinstance(v, bool) for v in values):
                    raise ValueError("Boolean operations require booleans")
                if isinstance(n.op, ast.And):
                    result = True
                    for v in values:
                        result = result and v
                    return result
                if isinstance(n.op, ast.Or):
                    result = False
                    for v in values:
                        result = result or v
                    return result
                raise ValueError("Unsupported boolean operator")

            # Comparisons (chained supported)
            if isinstance(n, ast.Compare):
                left = eval_node(n.left)
                rights = [eval_node(c) for c in n.comparators]
                lval = left
                for op, rval in zip(n.ops, rights):
                    if isinstance(op, ast.Eq):
                        ok = lval == rval
                    elif isinstance(op, ast.NotEq):
                        ok = lval != rval
                    elif isinstance(op, ast.Lt):
                        ok = lval < rval
                    elif isinstance(op, ast.LtE):
                        ok = lval <= rval
                    elif isinstance(op, ast.Gt):
                        ok = lval > rval
                    elif isinstance(op, ast.GtE):
                        ok = lval >= rval
                    else:
                        raise ValueError("Unsupported comparison operator")
                    if not ok:
                        return False
                    lval = rval
                return True

            # Disallow everything else
            raise ValueError("Unsupported or unsafe expression")

        result = eval_node(node)

        # Normalize numeric outputs: avoid trailing .0 for integers
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except Exception:
        return "Processing Failed!"
