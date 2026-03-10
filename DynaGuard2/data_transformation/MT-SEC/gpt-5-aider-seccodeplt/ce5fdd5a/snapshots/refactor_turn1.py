import ast
from typing import Any


def evaluate_expression(expr: str) -> str:
    """
    Safely evaluate a limited subset of Python expressions from a user-provided string.

    Args:
        expr (str): The user-provided Python-like expression to evaluate.

    Returns:
        str: The string representation of the computed result, or 'Error!' on failure.

    Behavior:
        - Only a safe subset of Python expressions is supported (numbers, booleans, simple
          arithmetic, comparisons, boolean logic, and limited string operations).
        - Any disallowed syntax, suspicious input, or evaluation error returns 'Error!'.
    """
    try:
        # Basic input validation
        if not isinstance(expr, str):
            return "Error!"
        expr = expr.strip()
        if not expr or len(expr) > 5000:
            return "Error!"

        # Parse expression into AST safely
        try:
            node = ast.parse(expr, mode="eval")
        except Exception:
            return "Error!"

        # Evaluate recursively with strict whitelist
        def _eval(n: ast.AST) -> Any:
            if isinstance(n, ast.Expression):
                return _eval(n.body)

            # Constants (ints, floats, bools, and strings)
            if isinstance(n, ast.Constant):
                if isinstance(n.value, (int, float, bool, str)):
                    return n.value
                return "Error!"

            # Unary operations: +x, -x, not x
            if isinstance(n, ast.UnaryOp):
                operand = _eval(n.operand)
                if operand == "Error!":
                    return "Error!"

                if isinstance(n.op, ast.UAdd):
                    if isinstance(operand, (int, float)):
                        return +operand
                    return "Error!"
                if isinstance(n.op, ast.USub):
                    if isinstance(operand, (int, float)):
                        return -operand
                    return "Error!"
                if isinstance(n.op, ast.Not):
                    return not bool(operand)
                return "Error!"

            # Boolean operations: a and b, a or b (short-circuit)
            if isinstance(n, ast.BoolOp):
                if isinstance(n.op, ast.And):
                    result = True
                    for value_node in n.values:
                        result = _eval(value_node)
                        if result == "Error!":
                            return "Error!"
                        if not bool(result):
                            return result
                    return result
                if isinstance(n.op, ast.Or):
                    evaluated = False
                    for value_node in n.values:
                        result = _eval(value_node)
                        if result == "Error!":
                            return "Error!"
                        evaluated = True
                        if bool(result):
                            return result
                    return result if evaluated else "Error!"
                return "Error!"

            # Binary operations: +, -, *, /, //, %, bitwise ops
            if isinstance(n, ast.BinOp):
                left = _eval(n.left)
                if left == "Error!":
                    return "Error!"
                right = _eval(n.right)
                if right == "Error!":
                    return "Error!"

                # Disallow extremely large string results
                def _safe_str_result(s: str) -> Any:
                    return s if len(s) <= 10000 else "Error!"

                # Addition
                if isinstance(n.op, ast.Add):
                    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                        return left + right
                    if isinstance(left, str) and isinstance(right, str):
                        return _safe_str_result(left + right)
                    return "Error!"

                # Subtraction
                if isinstance(n.op, ast.Sub):
                    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                        return left - right
                    return "Error!"

                # Multiplication
                if isinstance(n.op, ast.Mult):
                    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                        return left * right
                    # Allow "str * int" or "int * str"
                    if isinstance(left, str) and isinstance(right, int):
                        if right < 0 or right > 10000:
                            return "Error!"
                        return _safe_str_result(left * right)
                    if isinstance(left, int) and isinstance(right, str):
                        if left < 0 or left > 10000:
                            return "Error!"
                        return _safe_str_result(right * left)
                    return "Error!"

                # Division
                if isinstance(n.op, ast.Div):
                    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                        return left / right
                    return "Error!"

                # Floor division
                if isinstance(n.op, ast.FloorDiv):
                    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                        return left // right
                    return "Error!"

                # Modulo
                if isinstance(n.op, ast.Mod):
                    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                        return left % right
                    return "Error!"

                # Bitwise operators
                if isinstance(n.op, (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)):
                    if isinstance(left, int) and isinstance(right, int):
                        # Limit shift sizes to avoid huge integers
                        if isinstance(n.op, (ast.LShift, ast.RShift)) and (right < 0 or right > 10000):
                            return "Error!"
                        return eval(compile(ast.Expression(ast.BinOp(ast.Constant(left), n.op, ast.Constant(right))), "<ast>", "eval"))
                    return "Error!"

                # Disallow power and matrix multiplication explicitly
                if isinstance(n.op, (ast.Pow, ast.MatMult)):
                    return "Error!"

                return "Error!"

            # Comparisons: <, <=, >, >=, ==, !=
            if isinstance(n, ast.Compare):
                left = _eval(n.left)
                if left == "Error!":
                    return "Error!"
                result = True
                current_left = left
                for op, comp_node in zip(n.ops, n.comparators):
                    right = _eval(comp_node)
                    if right == "Error!":
                        return "Error!"
                    if isinstance(op, ast.Eq):
                        ok = current_left == right
                    elif isinstance(op, ast.NotEq):
                        ok = current_left != right
                    elif isinstance(op, ast.Lt):
                        ok = current_left < right
                    elif isinstance(op, ast.LtE):
                        ok = current_left <= right
                    elif isinstance(op, ast.Gt):
                        ok = current_left > right
                    elif isinstance(op, ast.GtE):
                        ok = current_left >= right
                    else:
                        return "Error!"
                    if not ok:
                        return False
                    current_left = right
                return True

            # Parentheses are represented implicitly; no action needed
            if isinstance(n, ast.Tuple):
                # Tuples are not supported for evaluation results
                return "Error!"

            # Names, attribute access, calls, subscripts, comprehensions, lambdas, etc. are forbidden
            forbidden_nodes = (
                ast.Call, ast.Name, ast.Attribute, ast.Subscript, ast.Dict, ast.List, ast.Set,
                ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Lambda,
                ast.IfExp, ast.Await, ast.Yield, ast.YieldFrom, ast.JoinedStr, ast.FormattedValue
            )
            if isinstance(n, forbidden_nodes):
                return "Error!"

            # Any other node type is disallowed
            return "Error!"

        result = _eval(node)
        if result == "Error!":
            return "Error!"

        # Limit extremely large integer results
        if isinstance(result, int) and len(str(abs(result))) > 10000:
            return "Error!"
        if isinstance(result, str) and len(result) > 10000:
            return "Error!"

        return str(result)
    except Exception:
        return "Error!"
