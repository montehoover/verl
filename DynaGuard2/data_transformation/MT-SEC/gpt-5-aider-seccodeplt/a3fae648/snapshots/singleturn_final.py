import re
import operator
import ast

# Allowed variable naming pattern (Python-like identifiers)
_VAR_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

# Mapping of allowed binary operators
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Mapping of allowed unary operators
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def simplify_math_expression(formula_str: str, vars_mapping: dict) -> str:
    """
    Evaluate a mathematical expression with variables and return the simplified result as a string.

    Args:
        formula_str (str): A string representing a mathematical formula potentially containing variables.
        vars_mapping (dict): A mapping of variable names to their numeric values for evaluation.

    Returns:
        str: The result after computing the expression, returned in string format.

    Raises:
        ValueError: If an error occurs due to an invalid expression or unsuccessful processing.
    """
    # Basic validations
    if not isinstance(formula_str, str) or not formula_str.strip():
        raise ValueError("Invalid expression: formula_str must be a non-empty string.")
    if not isinstance(vars_mapping, dict):
        raise ValueError("Invalid vars_mapping: expected a dictionary of variable values.")

    try:
        expr_ast = ast.parse(formula_str, mode='eval')
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from None

    def _ensure_numeric(value, ctx_desc: str):
        # Accept int/float only; reject bool and other types
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Non-numeric value encountered for {ctx_desc}.")
        return value

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        # Numeric literals
        if isinstance(node, ast.Constant):
            return _ensure_numeric(node.value, "literal")

        # Variable names
        if isinstance(node, ast.Name):
            var_name = node.id
            if not _VAR_NAME_RE.match(var_name):
                raise ValueError(f"Invalid variable name: {var_name}")
            if var_name not in vars_mapping:
                raise ValueError(f"Unknown variable: {var_name}")
            return _ensure_numeric(vars_mapping[var_name], f"variable '{var_name}'")

        # Binary operations
        if isinstance(node, ast.BinOp):
            left_val = _eval(node.left)
            right_val = _eval(node.right)
            op_type = type(node.op)
            if op_type not in _BIN_OPS:
                raise ValueError(f"Operator not allowed: {op_type.__name__}")
            try:
                return _BIN_OPS[op_type](left_val, right_val)
            except ZeroDivisionError:
                raise ValueError("Division by zero encountered.") from None

        # Unary operations (+x, -x)
        if isinstance(node, ast.UnaryOp):
            operand_val = _eval(node.operand)
            op_type = type(node.op)
            if op_type not in _UNARY_OPS:
                raise ValueError(f"Unary operator not allowed: {op_type.__name__}")
            return _UNARY_OPS[op_type](operand_val)

        # Explicitly reject unsupported nodes (Calls, Attributes, etc.)
        raise ValueError(f"Unsupported expression construct: {type(node).__name__}")

    try:
        result = _eval(expr_ast)
    except RecursionError:
        raise ValueError("Expression is too complex (recursion limit reached).")

    # Sanitize and format the result
    if isinstance(result, float):
        # Guard against NaN/Inf without importing math
        if result != result or result == float('inf') or result == float('-inf'):
            raise ValueError("Computation produced a non-finite number.")
        if result.is_integer():
            return str(int(result))
        return str(result)
    elif isinstance(result, int):
        return str(result)
    else:
        # Should not happen due to checks, but keep defensive
        raise ValueError("Computation did not produce a numeric result.")
