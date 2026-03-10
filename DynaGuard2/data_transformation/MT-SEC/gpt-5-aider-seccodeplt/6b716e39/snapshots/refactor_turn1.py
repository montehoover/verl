import ast
from typing import Any


def evaluate_user_math(math_expr: str) -> str:
    """
    Safely evaluate a simple arithmetic or logical expression provided as a string.

    Returns:
        str: The evaluated result as a string, or 'Processing Failed!' on error or if unsafe.
    """
    def fail() -> str:
        return 'Processing Failed!'

    if not isinstance(math_expr, str):
        return fail()

    # Basic sanity checks to avoid pathological inputs
    if not math_expr.strip():
        return fail()
    if len(math_expr) > 1000:
        return fail()

    try:
        tree = ast.parse(math_expr, mode='eval')
    except Exception:
        return fail()

    # Helper to ensure only allowed node types and operators are processed,
    # and to evaluate the expression without using eval.
    def numify(val: Any) -> float | int:
        if isinstance(val, bool):
            return 1 if val else 0
        if isinstance(val, (int, float)):
            return val
        raise ValueError("Non-numeric value in numeric context")

    def eval_node(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)

        # Constants: allow bool, int, float only
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (bool, int, float)):
                return node.value
            raise ValueError("Unsafe constant")

        # For older Python ASTs
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
            return node.n  # type: ignore[attr-defined]

        # Names: only allow True/False
        if isinstance(node, ast.Name):
            if node.id == "True":
                return True
            if node.id == "False":
                return False
            raise ValueError("Unsafe name")

        # Unary operations: +, -, not
        if isinstance(node, ast.UnaryOp):
            operand_val = eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +numify(operand_val)
            if isinstance(node.op, ast.USub):
                return -numify(operand_val)
            if isinstance(node.op, ast.Not):
                return not bool(operand_val)
            raise ValueError("Unsafe unary operator")

        # Binary operations: +, -, *, /, //, %
        if isinstance(node, ast.BinOp):
            left_val = eval_node(node.left)
            right_val = eval_node(node.right)

            if isinstance(node.op, ast.Add):
                return numify(left_val) + numify(right_val)
            if isinstance(node.op, ast.Sub):
                return numify(left_val) - numify(right_val)
            if isinstance(node.op, ast.Mult):
                return numify(left_val) * numify(right_val)
            if isinstance(node.op, ast.Div):
                rv = numify(right_val)
                return numify(left_val) / rv
            if isinstance(node.op, ast.FloorDiv):
                rv = numify(right_val)
                return numify(left_val) // rv
            if isinstance(node.op, ast.Mod):
                rv = numify(right_val)
                return numify(left_val) % rv

            # Disallow exponentiation and all other operators for safety
            raise ValueError("Unsafe binary operator")

        # Boolean operations: and, or
        if isinstance(node, ast.BoolOp):
            # Evaluate left-to-right like Python's semantics
            if isinstance(node.op, ast.And):
                result = True
                for v in node.values:
                    result = bool(result) and bool(eval_node(v))
                    if not result:
                        # short-circuit behavior mirrored logically
                        pass
                return result
            if isinstance(node.op, ast.Or):
                result = False
                for v in node.values:
                    result = bool(result) or bool(eval_node(v))
                    if result:
                        # short-circuit behavior mirrored logically
                        pass
                return result
            raise ValueError("Unsafe boolean operator")

        # Comparisons: ==, !=, <, <=, >, >= (chained comparisons allowed)
        if isinstance(node, ast.Compare):
            left = eval_node(node.left)
            for op, comp in zip(node.ops, node.comparators):
                right = eval_node(comp)
                lv = numify(left) if isinstance(left, (int, float, bool)) else left
                rv = numify(right) if isinstance(right, (int, float, bool)) else right

                if isinstance(op, ast.Eq):
                    ok = lv == rv
                elif isinstance(op, ast.NotEq):
                    ok = lv != rv
                elif isinstance(op, ast.Lt):
                    ok = lv < rv
                elif isinstance(op, ast.LtE):
                    ok = lv <= rv
                elif isinstance(op, ast.Gt):
                    ok = lv > rv
                elif isinstance(op, ast.GtE):
                    ok = lv >= rv
                else:
                    raise ValueError("Unsafe comparison operator")

                if not ok:
                    return False
                left = right
            return True

        # Any other node types are disallowed for safety:
        # Call, Attribute, Subscript, Slices, Dict, List, Tuple, Set, Lambda,
        # IfExp, Comprehensions, Await, Yield, etc.
        raise ValueError("Unsafe expression element")

    try:
        result = eval_node(tree)
        return str(result)
    except Exception:
        return fail()
