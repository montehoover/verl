import ast
import operator
from typing import Any


_ALLOWED_BINOPS = (
    ast.Add,  # +
    ast.Sub,  # -
    ast.Mult,  # *
    ast.Div,  # /
    ast.FloorDiv,  # //
    ast.Mod,  # %
    ast.Pow,  # **
)

_ALLOWED_UNARYOPS = (
    ast.UAdd,  # +
    ast.USub,  # -
    ast.Not,   # not
)

_ALLOWED_BOOLOPS = (
    ast.And,  # and
    ast.Or,   # or
)

_ALLOWED_CMPOPS = (
    ast.Eq,   # ==
    ast.NotEq,  # !=
    ast.Lt,   # <
    ast.LtE,  # <=
    ast.Gt,   # >
    ast.GtE,  # >=
)


def _is_safe_node(node: ast.AST) -> bool:
    # Entry node can be Expression in 'eval' mode, but we'll always pass body in parse_expression.
    if isinstance(node, ast.BinOp):
        return (
            isinstance(node.op, _ALLOWED_BINOPS)
            and _is_safe_node(node.left)
            and _is_safe_node(node.right)
        )

    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, _ALLOWED_UNARYOPS) and _is_safe_node(node.operand)

    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, _ALLOWED_BOOLOPS):
            return False
        return all(_is_safe_node(v) for v in node.values)

    if isinstance(node, ast.Compare):
        # Allow chained comparisons like 1 < x <= 3 (if names were allowed; here only constants)
        if not all(isinstance(op, _ALLOWED_CMPOPS) for op in node.ops):
            return False
        if not _is_safe_node(node.left):
            return False
        return all(_is_safe_node(comp) for comp in node.comparators)

    # Constants: allow numeric and boolean literals only
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float, bool))

    # For older Python versions where booleans may appear as NameConstant
    if hasattr(ast, "NameConstant") and isinstance(node, getattr(ast, "NameConstant")):
        return node.value in (True, False)

    # Disallow tuples/lists/sets/dicts, calls, attributes, subscripts, comprehensions, lambdas, etc.
    # Explicitly block Name usage (variables), except possibly "True"/"False" which should be Constant in modern Python.
    if isinstance(node, ast.Name):
        return node.id in ("True", "False")

    # Parenthesized expressions do not create a special node; no explicit handling needed.

    return False


def parse_expression(expr: str) -> bool:
    """
    Validate a user-supplied expression for safety and allowed operations.

    Allowed:
      - Arithmetic: +, -, *, /, //, %, **
      - Unary: +x, -x, not x
      - Logical: and, or
      - Comparisons: ==, !=, <, <=, >, >=
      - Literals: integers, floats, booleans

    Disallowed:
      - Any names/variables (except boolean literals), function calls, attribute access,
        subscripts, comprehensions, lambdas, if-expressions, imports, etc.

    Returns True if the expression parses and only contains allowed operations and literals.
    """
    if not isinstance(expr, str):
        return False

    # Optional guard against extremely large inputs
    if len(expr) > 10000:
        return False

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False
    except Exception:
        return False

    try:
        return _is_safe_node(tree.body)
    except RecursionError:
        # Extremely deep/recursive structures are considered invalid
        return False


# ---- Evaluation ----

_BINOP_IMPL = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_CMPOP_IMPL = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def evaluate_safe_expression(expr: str) -> Any:
    """
    Evaluate a validated expression string consisting only of allowed arithmetic and logical operations.

    Returns:
      - The computed value if the expression is valid and evaluation succeeds.
      - A string error message indicating a potential safety violation if disallowed operations are encountered.
      - A string error message for syntax or evaluation errors.
    """
    if not isinstance(expr, str):
        return "Error: expression must be a string."

    if len(expr) > 10000:
        return "Error: expression too long."

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return f"Error: invalid expression syntax ({e})."
    except Exception as e:
        return f"Error: failed to parse expression ({e})."

    body = tree.body

    # Validate safety before evaluation
    try:
        if not _is_safe_node(body):
            return "Potential safety violation: expression contains disallowed operations or values."
    except RecursionError:
        return "Error: expression too deeply nested."

    def _eval_node(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, (int, float, bool)):
                return val
            raise ValueError("Potential safety violation: disallowed constant type.")

        if isinstance(node, ast.Name):
            if node.id == "True":
                return True
            if node.id == "False":
                return False
            raise ValueError("Potential safety violation: disallowed name usage.")

        if isinstance(node, ast.UnaryOp):
            op = node.op
            operand = _eval_node(node.operand)
            if isinstance(op, ast.UAdd):
                return +operand
            if isinstance(op, ast.USub):
                return -operand
            if isinstance(op, ast.Not):
                return not operand
            raise ValueError("Potential safety violation: disallowed unary operation.")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _BINOP_IMPL:
                raise ValueError("Potential safety violation: disallowed binary operation.")
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            return _BINOP_IMPL[op_type](left, right)

        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                # Short-circuit AND
                result = _eval_node(node.values[0])
                if not result:
                    return result
                for v in node.values[1:]:
                    result = _eval_node(v)
                    if not result:
                        return result
                return result
            if isinstance(node.op, ast.Or):
                # Short-circuit OR
                result = _eval_node(node.values[0])
                if result:
                    return result
                for v in node.values[1:]:
                    result = _eval_node(v)
                    if result:
                        return result
                return result
            raise ValueError("Potential safety violation: disallowed boolean operation.")

        if isinstance(node, ast.Compare):
            left = _eval_node(node.left)
            for op, comp in zip(node.ops, node.comparators):
                right = _eval_node(comp)
                op_type = type(op)
                func = _CMPOP_IMPL.get(op_type)
                if func is None:
                    raise ValueError("Potential safety violation: disallowed comparison operation.")
                if not func(left, right):
                    return False
                left = right
            return True

        # Any other node types are disallowed
        raise ValueError("Potential safety violation: disallowed syntax or operation.")

    try:
        return _eval_node(body)
    except ZeroDivisionError:
        return "Error: division by zero."
    except OverflowError:
        return "Error: numeric overflow."
    except RecursionError:
        return "Error: expression too deeply nested."
    except ValueError as ve:
        # Our own safety violation messages
        msg = str(ve)
        if msg.startswith("Potential safety violation"):
            return msg
        return f"Error: {msg}"
    except Exception as e:
        return f"Error: evaluation failed ({e})."


def calculate_user_input(input_expr: str) -> Any:
    """
    Securely process and evaluate a user-supplied expression.

    - Returns the computed result for valid arithmetic/logical expressions.
    - Returns 'Processing Failed!' if the expression is unsafe, invalid, or evaluation fails.
    """
    if not isinstance(input_expr, str):
        return "Processing Failed!"

    # Basic size guard to avoid pathological inputs
    if len(input_expr) > 10000:
        return "Processing Failed!"

    try:
        # Validate the expression structure first
        if not parse_expression(input_expr):
            return "Processing Failed!"

        # Evaluate safely
        result = evaluate_safe_expression(input_expr)

        # If evaluation returned an error message, treat as failure
        if isinstance(result, str):
            return "Processing Failed!"

        return result
    except Exception:
        # Any unexpected issues are treated as a processing failure
        return "Processing Failed!"
