import re
import operator
import ast

# Supported operators
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Supported node types
_SUPPORTED_NODES = (
    ast.Expression,
    ast.Num,  # Deprecated in Python 3.8, use ast.Constant
    ast.Constant,
    ast.Name,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call, # For functions if we extend later, but not for basic arithmetic
)


def _evaluate_node(node, var_dict):
    """
    Recursively evaluates an AST node.
    """
    if isinstance(node, ast.Num):  # For older Python versions
        return node.n
    if isinstance(node, ast.Constant): # For Python 3.8+
        return node.value
    elif isinstance(node, ast.Name):
        if node.id in var_dict:
            return var_dict[node.id]
        else:
            raise ValueError(f"Undefined variable: {node.id}")
    elif isinstance(node, ast.BinOp):
        left_val = _evaluate_node(node.left, var_dict)
        right_val = _evaluate_node(node.right, var_dict)
        op_func = _OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        try:
            return op_func(left_val, right_val)
        except ZeroDivisionError:
            raise ValueError("Division by zero")
        except TypeError:
            raise ValueError(f"Invalid types for operator {type(node.op).__name__}: {type(left_val).__name__}, {type(right_val).__name__}")
    elif isinstance(node, ast.UnaryOp):
        operand_val = _evaluate_node(node.operand, var_dict)
        op_func = _OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        try:
            return op_func(operand_val)
        except TypeError:
            raise ValueError(f"Invalid type for operator {type(node.op).__name__}: {type(operand_val).__name__}")
    elif isinstance(node, ast.Expression):
        return _evaluate_node(node.body, var_dict)
    else:
        # Check if any part of the expression uses unsupported nodes
        for sub_node in ast.walk(node):
            if not isinstance(sub_node, _SUPPORTED_NODES):
                 raise ValueError(f"Unsupported expression component: {type(sub_node).__name__}")
        # If all sub-nodes are supported but we reached here, it's an unhandled structure
        raise ValueError(f"Unsupported expression structure: {type(node).__name__}")


def execute_calculation_string(calc_expr: str, var_dict: dict = None) -> str:
    """
    Computes and simplifies a mathematical expression string with variable substitution.

    Args:
        calc_expr: A string representing the mathematical expression.
        var_dict: A dictionary for variable substitution, where keys are
                  variable names (strings) and values are their
                  corresponding numbers.

    Returns:
        The result of the computation as a string.

    Raises:
        ValueError: If the input expression is invalid, contains unsupported
                    operations/syntax, or if the computation fails (e.g.,
                    division by zero, undefined variable).
    """
    if var_dict is None:
        var_dict = {}

    if not isinstance(calc_expr, str):
        raise ValueError("Expression must be a string.")
    if not isinstance(var_dict, dict):
        raise ValueError("Variable dictionary must be a dictionary.")

    for key, value in var_dict.items():
        if not isinstance(key, str):
            raise ValueError(f"Variable names must be strings: {key}")
        if not isinstance(value, (int, float)):
            raise ValueError(f"Variable values must be numbers: {key}={value}")

    try:
        # Sanitize expression: allow only basic math, numbers, variables, parentheses
        # This regex is a basic check; AST parsing is the main security layer.
        if not re.fullmatch(r"[a-zA-Z0-9_+\-*/().\s]+", calc_expr):
            raise ValueError("Expression contains invalid characters.")

        # Parse the expression string into an AST
        # Mode 'eval' is used for a single expression
        parsed_ast = ast.parse(calc_expr, mode='eval')

        # Validate AST nodes (optional, _evaluate_node does this implicitly)
        for node in ast.walk(parsed_ast):
            if not isinstance(node, _SUPPORTED_NODES):
                raise ValueError(f"Unsupported syntax or operation: {type(node).__name__}")

        # Evaluate the AST
        result = _evaluate_node(parsed_ast, var_dict)

        if not isinstance(result, (int, float)):
             raise ValueError(f"Expression did not evaluate to a number: {result}")

        return str(result)
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax in expression: {calc_expr}. Error: {e}")
    except ValueError as e: # Catch ValueErrors from _evaluate_node or our checks
        raise e
    except Exception as e:
        # Catch any other unexpected errors during parsing or evaluation
        raise ValueError(f"An unexpected error occurred: {e}")
