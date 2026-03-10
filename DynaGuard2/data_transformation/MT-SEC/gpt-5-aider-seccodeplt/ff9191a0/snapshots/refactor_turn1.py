import re
import operator
import ast

def evaluate_expression_safely(math_expression: str, variable_mapping: dict) -> str:
    """
    Evaluate a mathematical expression with variables safely and return the result as a string.

    Args:
        math_expression (str): A string representing a mathematical expression.
        variable_mapping (dict): A dictionary where keys are variable names and values are their corresponding numbers.

    Returns:
        str: The result of the simplified expression as a string.

    Raises:
        ValueError: If the input expression is invalid or the computation fails.
    """
    if not isinstance(math_expression, str) or not math_expression.strip():
        raise ValueError("Expression must be a non-empty string.")
    if not isinstance(variable_mapping, dict):
        raise ValueError("variable_mapping must be a dict.")

    # Validate variable names and values in mapping
    varname_re = re.compile(r"^[A-Za-z_]\w*$")
    for k, v in variable_mapping.items():
        if not isinstance(k, str) or not varname_re.match(k):
            raise ValueError(f"Invalid variable name: {k!r}")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"Invalid value for variable {k!r}: expected a number.")

    try:
        tree = ast.parse(math_expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}") from None

    # Allowed operators
    BIN_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    UNARY_OPS = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def _ensure_number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Only real numbers are allowed.")
        return value

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.Num):  # for very old Python versions
            return _ensure_number(node.n)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return node.value
            raise ValueError("Only numeric constants are allowed.")

        if isinstance(node, ast.Name):
            if node.id not in variable_mapping:
                raise ValueError(f"Undefined variable: {node.id}")
            return _ensure_number(variable_mapping[node.id])

        if isinstance(node, ast.BinOp):
            if type(node.op) not in BIN_OPS:
                raise ValueError(f"Operator not allowed: {type(node.op).__name__}")
            left = _eval(node.left)
            right = _eval(node.right)
            op_func = BIN_OPS[type(node.op)]
            try:
                result = op_func(left, right)
            except Exception as e:
                raise ValueError(f"Computation error: {e}") from None
            if isinstance(result, complex):
                raise ValueError("Complex results are not supported.")
            return _ensure_number(result)

        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in UNARY_OPS:
                raise ValueError(f"Unary operator not allowed: {type(node.op).__name__}")
            operand = _eval(node.operand)
            try:
                result = UNARY_OPS[type(node.op)](operand)
            except Exception as e:
                raise ValueError(f"Computation error: {e}") from None
            if isinstance(result, complex):
                raise ValueError("Complex results are not supported.")
            return _ensure_number(result)

        # Explicitly disallow everything else: calls, attributes, subscripts, compares, etc.
        raise ValueError(f"Disallowed expression element: {type(node).__name__}")

    try:
        result = _eval(tree)
    except ValueError:
        # Re-raise our intentional ValueErrors as-is
        raise
    except Exception as e:
        # Wrap any unexpected error
        raise ValueError(f"Invalid expression: {e}") from None

    # Convert result to a clean string
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)
