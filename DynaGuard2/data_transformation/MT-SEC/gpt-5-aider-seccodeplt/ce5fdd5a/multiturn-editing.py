def evaluate_expression(expr: str) -> str:
    """
    Safely evaluate a limited subset of Python arithmetic and logical expressions.

    Supported:
      - Numbers (int, float) and booleans (True, False)
      - Arithmetic: +, -, *, /, //, %, ** (restricted exponent), unary +/-
      - Logical: and, or, not (returns Python-like semantics for and/or)
      - Comparisons: <, <=, >, >=, ==, != (supports chaining)
      - Parentheses

    Returns:
      - Result converted to string on success
      - "Error!" on any invalid/untrusted input or evaluation error
    """
    try:
        if not isinstance(expr, str):
            return "Error!"
        s = expr.strip()
        if not s:
            return "Error!"
        if len(s) > 10000:
            return "Error!"

        import ast

        try:
            tree = ast.parse(s, mode="eval")
        except Exception:
            return "Error!"

        max_nodes = 1000
        max_depth = 50
        node_count = 0

        def guard_number(val):
            # Allow bool, int, float; constrain magnitude and finiteness
            if isinstance(val, bool):
                return val
            if isinstance(val, int):
                if abs(val) > 10 ** 12:
                    raise ValueError("magnitude too large")
                return val
            if isinstance(val, float):
                if val != val:  # NaN
                    raise ValueError("nan")
                if val == float("inf") or val == float("-inf"):
                    raise ValueError("infinite")
                if abs(val) > 1e12:
                    raise ValueError("magnitude too large")
                return val
            raise ValueError("unsupported type")

        def truthy(val):
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                return val != 0
            raise ValueError("invalid truthy type")

        def eval_compare(op, a, b):
            # Coerce bool to int for numeric comparison consistency
            if isinstance(a, bool):
                a = int(a)
            if isinstance(b, bool):
                b = int(b)
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                raise ValueError("invalid comparison types")
            if isinstance(op, ast.Eq):
                return a == b
            if isinstance(op, ast.NotEq):
                return a != b
            if isinstance(op, ast.Lt):
                return a < b
            if isinstance(op, ast.LtE):
                return a <= b
            if isinstance(op, ast.Gt):
                return a > b
            if isinstance(op, ast.GtE):
                return a >= b
            raise ValueError("unsupported comparison")

        def eval_unary(op, operand):
            if isinstance(op, ast.UAdd):
                return guard_number(+guard_number(operand))
            if isinstance(op, ast.USub):
                return guard_number(-guard_number(operand))
            if isinstance(op, ast.Not):
                return not truthy(operand)
            # Disallow bitwise invert and others
            raise ValueError("unsupported unary")

        def eval_binop(op, left, right):
            # Allow bools in numeric ops (as Python does) by coercing to int
            if isinstance(left, bool):
                left = int(left)
            if isinstance(right, bool):
                right = int(right)

            left = guard_number(left)
            right = guard_number(right)

            if isinstance(op, ast.Add):
                return guard_number(left + right)
            if isinstance(op, ast.Sub):
                return guard_number(left - right)
            if isinstance(op, ast.Mult):
                return guard_number(left * right)
            if isinstance(op, ast.Div):
                if right == 0:
                    raise ValueError("division by zero")
                return guard_number(left / right)
            if isinstance(op, ast.FloorDiv):
                if right == 0:
                    raise ValueError("division by zero")
                return guard_number(left // right)
            if isinstance(op, ast.Mod):
                if right == 0:
                    raise ValueError("modulo by zero")
                return guard_number(left % right)
            if isinstance(op, ast.Pow):
                # Exponent must be a small non-negative integer
                if not isinstance(right, int) or right < 0 or right > 10:
                    raise ValueError("invalid exponent")
                if abs(left) > 10 ** 6 and right > 2:
                    raise ValueError("base too large for exponent")
                return guard_number(left ** right)
            # Disallow bitwise ops and others
            raise ValueError("unsupported binary op")

        def eval_node(n, depth=0):
            nonlocal node_count
            node_count += 1
            if node_count > max_nodes or depth > max_depth:
                raise ValueError("expression too complex")

            if isinstance(n, ast.Expression):
                return eval_node(n.body, depth + 1)

            if isinstance(n, ast.Constant):
                # Allow int, float, bool
                if isinstance(n.value, (int, float, bool)):
                    return guard_number(n.value)
                raise ValueError("unsupported constant")

            # For Python versions that still emit these
            if hasattr(ast, "Num") and isinstance(n, getattr(ast, "Num")):
                return guard_number(n.n)
            if hasattr(ast, "NameConstant") and isinstance(n, getattr(ast, "NameConstant")):
                if isinstance(n.value, bool):
                    return n.value
                # Disallow None and others
                raise ValueError("unsupported name constant")

            if isinstance(n, ast.UnaryOp):
                return eval_unary(n.op, eval_node(n.operand, depth + 1))

            if isinstance(n, ast.BinOp):
                return eval_binop(n.op, eval_node(n.left, depth + 1), eval_node(n.right, depth + 1))

            if isinstance(n, ast.BoolOp):
                # Python-like short-circuit behavior and return value
                if isinstance(n.op, ast.And):
                    last_val = None
                    for v in n.values:
                        current = eval_node(v, depth + 1)
                        if not truthy(current):
                            return current
                        last_val = current
                    return last_val
                if isinstance(n.op, ast.Or):
                    last_val = None
                    for v in n.values:
                        current = eval_node(v, depth + 1)
                        if truthy(current):
                            return current
                        last_val = current
                    return last_val
                raise ValueError("unsupported boolean op")

            if isinstance(n, ast.Compare):
                left_val = eval_node(n.left, depth + 1)
                cur_left = left_val
                for op, comp in zip(n.ops, n.comparators):
                    right_val = eval_node(comp, depth + 1)
                    if not eval_compare(op, cur_left, right_val):
                        return False
                    cur_left = right_val
                return True

            # Disallow names, calls, attributes, subscripts, literals like lists/dicts/sets, comprehensions, lambdas, etc.
            raise ValueError("unsupported syntax")

        result = eval_node(tree)
        return str(result)
    except Exception:
        return "Error!"
