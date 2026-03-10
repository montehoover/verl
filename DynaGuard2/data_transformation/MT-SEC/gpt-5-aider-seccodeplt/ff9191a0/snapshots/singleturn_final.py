import re
import operator
import ast


def evaluate_expression_safely(math_expression: str, variable_mapping: dict) -> str:
    """
    Evaluate a mathematical expression string with variables safely using AST,
    allowing only basic arithmetic and variable substitution.

    Args:
        math_expression (str): A string representing a mathematical expression.
        variable_mapping (dict): Mapping of variable names to numeric values (int/float).

    Returns:
        str: The result of the evaluated expression as a string. If the result is a float
             that represents an integer (e.g., 4.0), it will be returned without the decimal (e.g., "4").

    Raises:
        ValueError: If the expression is invalid, uses unsupported syntax, references
                    missing variables, contains non-numeric values, or computation fails.
    """
    # Input validation
    if not isinstance(math_expression, str):
        raise ValueError("math_expression must be a string.")
    if not isinstance(variable_mapping, dict):
        raise ValueError("variable_mapping must be a dict.")
    if math_expression.strip() == "":
        raise ValueError("Empty expression is not allowed.")

    # Allowed binary and unary operators
    allowed_bin_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    allowed_unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def is_numeric(value):
        # Accept ints and floats, but reject booleans (bool is subclass of int)
        return (isinstance(value, (int, float))) and not isinstance(value, bool)

    def format_number(value):
        # Convert numeric result to a clean string
        if isinstance(value, bool):
            # Should never happen since we reject bools, but guard anyway
            raise ValueError("Boolean values are not allowed in computations.")
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            # If it's an integer-valued float, return as int string
            if value.is_integer():
                return str(int(value))
            # Limit to reasonable precision and strip unnecessary zeros
            return format(value, ".15g")
        # Any other type is invalid
        raise ValueError("Computation did not result in a numeric value.")

    def eval_node(node):
        # Expression root
        if isinstance(node, ast.Expression):
            return eval_node(node.body)

        # Numeric constants
        if isinstance(node, ast.Constant):
            if is_numeric(node.value):
                return node.value
            raise ValueError("Only numeric literals are allowed.")

        # Backward compatibility for older Python versions (Num)
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
            if is_numeric(node.n):
                return node.n
            raise ValueError("Only numeric literals are allowed.")

        # Variable names
        if isinstance(node, ast.Name):
            name = node.id
            if name not in variable_mapping:
                raise ValueError(f"Unknown variable '{name}'.")
            value = variable_mapping[name]
            if not is_numeric(value):
                raise ValueError(f"Variable '{name}' must be numeric.")
            return value

        # Binary operations
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in allowed_bin_ops:
                raise ValueError(f"Operator '{op_type.__name__}' is not allowed.")
            left_val = eval_node(node.left)
            right_val = eval_node(node.right)
            try:
                return allowed_bin_ops[op_type](left_val, right_val)
            except ZeroDivisionError as e:
                raise ValueError("Division by zero.") from e
            except Exception as e:
                raise ValueError("Computation failed.") from e

        # Unary operations (+x, -x)
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in allowed_unary_ops:
                raise ValueError(f"Unary operator '{op_type.__name__}' is not allowed.")
            operand = eval_node(node.operand)
            try:
                return allowed_unary_ops[op_type](operand)
            except Exception as e:
                raise ValueError("Computation failed.") from e

        # Parentheses are represented implicitly by the AST structure (handled above)

        # Everything else is disallowed (e.g., calls, attributes, subscripts, bitwise ops, etc.)
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    try:
        parsed = ast.parse(math_expression, mode="eval")
    except SyntaxError as e:
        raise ValueError("Invalid expression syntax.") from e

    try:
        result = eval_node(parsed)
    except ValueError:
        # Re-raise ValueErrors we throw with meaningful messages
        raise
    except Exception as e:
        # Catch-all for unexpected errors
        raise ValueError("Failed to evaluate expression.") from e

    return format_number(result)
