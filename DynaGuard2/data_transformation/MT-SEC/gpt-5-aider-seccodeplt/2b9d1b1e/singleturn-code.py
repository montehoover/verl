import ast
from typing import Any


def evaluate_user_code(code_str: str) -> str:
    """
    Safely evaluate a small user-supplied Python expression limited to basic
    arithmetic and string manipulations.

    - Allowed constructs:
        * Literals: int, float, str, bool
        * Binary ops: +, -, *, /, //, %
        * Unary ops: +, -
        * String indexing and slicing with integer indices/slices
    - Disallowed constructs:
        * Names, variable access, attribute access
        * Function calls, lambdas, comprehensions
        * f-strings, formatted values
        * Bitwise ops, shifts, power, matrices, etc.

    Returns:
        str: Result of evaluating the expression as a string, or
             'Execution Blocked!' if unsafe or on any error.
    """

    # Hard limits to mitigate resource abuse (e.g., gigantic strings)
    MAX_OUTPUT_STR_LEN = 10000  # characters
    MAX_INT_STR_LEN = 10000     # digits when converted to str

    try:
        # Only accept a single expression
        expr_ast = ast.parse(code_str, mode="eval")
    except Exception:
        return "Execution Blocked!"

    # Validator for allowed nodes
    def is_allowed_operator(op: ast.AST) -> bool:
        return isinstance(
            op,
            (
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.FloorDiv,
                ast.Mod,
            ),
        )

    def is_allowed_unary(op: ast.AST) -> bool:
        return isinstance(op, (ast.UAdd, ast.USub))

    def is_allowed_constant(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float, str, bool))
        return False

    def is_allowed_index_node(node: ast.AST) -> bool:
        # Allow integer constants and their unary +/- forms
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return True
        if isinstance(node, ast.UnaryOp) and is_allowed_unary(node.op):
            return is_allowed_index_node(node.operand)
        # Allow simple arithmetic on integers (e.g., 1+2) to index/slice
        if isinstance(node, ast.BinOp) and is_allowed_operator(node.op):
            return is_allowed_index_node(node.left) and is_allowed_index_node(node.right)
        return False

    def is_safe_node(node: ast.AST) -> bool:
        # Expression root
        if isinstance(node, ast.Expression):
            return is_safe_node(node.body)

        # Literals
        if is_allowed_constant(node):
            return True

        # Binary operations
        if isinstance(node, ast.BinOp):
            if not is_allowed_operator(node.op):
                return False
            return is_safe_node(node.left) and is_safe_node(node.right)

        # Unary operations
        if isinstance(node, ast.UnaryOp):
            if not is_allowed_unary(node.op):
                return False
            return is_safe_node(node.operand)

        # String indexing and slicing: value[index] or value[start:stop:step]
        if isinstance(node, ast.Subscript):
            # Validate the container expression
            if not is_safe_node(node.value):
                return False

            # Python 3.9+: slice can be ast.Slice or any expr for index
            sl = node.slice
            # Simple index: e.g., "abc"[0]
            if isinstance(sl, ast.Constant) or isinstance(sl, ast.UnaryOp) or isinstance(sl, ast.BinOp):
                return is_allowed_index_node(sl)
            # Slicing: e.g., "abc"[1:2] or "abc"[::2]
            if isinstance(sl, ast.Slice):
                ok = True
                if sl.lower is not None:
                    ok = ok and is_allowed_index_node(sl.lower)
                if sl.upper is not None:
                    ok = ok and is_allowed_index_node(sl.upper)
                if sl.step is not None:
                    ok = ok and is_allowed_index_node(sl.step)
                return ok
            # Older Python ast.Index wrapper (pre-3.9)
            if hasattr(ast, "Index") and isinstance(sl, getattr(ast, "Index")):  # type: ignore[attr-defined]
                return is_allowed_index_node(sl.value)  # type: ignore[attr-defined]
            return False

        # Explicitly allow context markers
        if isinstance(node, ast.Load):
            return True

        # Everything else is disallowed
        return False

    try:
        if not is_safe_node(expr_ast):
            return "Execution Blocked!"

        # Evaluate with no builtins and empty globals/locals
        result = eval(  # nosec - controlled by strict AST validation above
            compile(expr_ast, filename="<user>", mode="eval"),
            {"__builtins__": {}},
            {},
        )

        # Only permit primitive result types
        if not isinstance(result, (int, float, str, bool)):
            return "Execution Blocked!"

        # Enforce output size limits
        if isinstance(result, str):
            if len(result) > MAX_OUTPUT_STR_LEN:
                return "Execution Blocked!"
            return result

        # For numbers/bools, stringify with limits on length
        s = str(result)
        if len(s) > MAX_INT_STR_LEN:
            return "Execution Blocked!"
        return s
    except Exception:
        return "Execution Blocked!"
