import ast
import operator
from numbers import Real

_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def execute_operation(operation: str):
    """
    Execute a simple arithmetic program from a string and return the final result.

    Supported:
      - Numbers (int, float)
      - Variables (assignment to simple names, usage in expressions)
      - Binary ops: +, -, *, /, //, %, **
      - Unary ops: +, -
      - Parentheses
      - Multiple statements separated by semicolons or newlines

    Semantics:
      - The input is parsed as a sequence of statements.
      - Assignments set variables in an internal environment.
      - The "final result" is the value of the last evaluated statement:
          * For an assignment, it's the assigned value.
          * For a bare expression, it's that expression's value.

    Parameters:
        operation (str): e.g., "a = 5; b = a * 2" or "(1 + 2) * 3"

    Returns:
        int or float: The final computed value.

    Raises:
        TypeError: If operation is not a string.
        ValueError: If the program contains unsupported syntax or is empty.
        NameError: If a variable is used before assignment.
        ZeroDivisionError: If a division by zero occurs.
    """
    if not isinstance(operation, str):
        raise TypeError("operation must be a string")

    try:
        tree = ast.parse(operation, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Invalid program: {e}") from e

    env: dict[str, Real] = {}
    last_value: Real | None = None

    def _ensure_number(value):
        if isinstance(value, bool):
            # Disallow booleans even though they are ints in Python
            raise ValueError("Only int and float values are allowed")
        if not isinstance(value, Real):
            raise ValueError("Only numeric values (int or float) are allowed")
        return value

    def _eval_expr(node):
        # Numeric literal
        if isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return val
            raise ValueError("Only int and float constants are allowed")

        # For Python versions where numbers may appear as ast.Num
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
            val = node.n  # type: ignore[no-any-return]
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return val
            raise ValueError("Only int and float constants are allowed")

        # Variable usage
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            raise NameError(f"Undefined variable: {node.id}")

        # Unary operations (+x, -x)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
            operand = _eval_expr(node.operand)
            _ensure_number(operand)
            return _ALLOWED_UNARY_OPS[type(node.op)](operand)

        # Binary operations (x + y, x * y, etc.)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
            left = _eval_expr(node.left)
            right = _eval_expr(node.right)
            _ensure_number(left)
            _ensure_number(right)
            func = _ALLOWED_BIN_OPS[type(node.op)]
            return func(left, right)

        # Parentheses are represented implicitly in the AST by nested nodes; no special handling needed.

        # Disallow all other constructs (calls, attributes, subscripts, comprehensions, etc.)
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    def _exec_stmt(stmt):
        nonlocal last_value

        # Expression statement: evaluate and record as last value
        if isinstance(stmt, ast.Expr):
            last_value = _eval_expr(stmt.value)
            return

        # Simple assignment: name = <expr>
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise ValueError("Only simple assignments to a single variable are allowed")
            target = stmt.targets[0].id
            value = _eval_expr(stmt.value)
            _ensure_number(value)
            env[target] = value
            last_value = value
            return

        # Augmented assignment: name += <expr>, etc.
        if isinstance(stmt, ast.AugAssign):
            if not isinstance(stmt.target, ast.Name):
                raise ValueError("Only augmented assignments to a single variable are allowed")
            name = stmt.target.id
            if name not in env:
                raise NameError(f"Undefined variable: {name}")
            if type(stmt.op) not in _ALLOWED_BIN_OPS:
                raise ValueError(f"Unsupported operator in augmented assignment: {type(stmt.op).__name__}")
            right = _eval_expr(stmt.value)
            _ensure_number(right)
            func = _ALLOWED_BIN_OPS[type(stmt.op)]
            new_val = func(env[name], right)
            _ensure_number(new_val)
            env[name] = new_val
            last_value = new_val
            return

        # Disallow control flow, imports, function defs, etc.
        raise ValueError(f"Unsupported statement: {type(stmt).__name__}")

    # Process each statement in order
    if not isinstance(tree, ast.Module):
        raise ValueError("Invalid program structure")

    for stmt in tree.body:
        _exec_stmt(stmt)

    if last_value is None:
        raise ValueError("No evaluatable statements found")

    return last_value
