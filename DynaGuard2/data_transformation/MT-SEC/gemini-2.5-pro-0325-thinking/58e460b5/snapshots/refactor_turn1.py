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
    ast.UAdd: operator.pos, # For completeness, though +5 is usually just ast.Constant(5)
}

def _eval_expr(node, var_dict):
    """
    Recursively evaluates an AST node.
    """
    if isinstance(node, ast.Constant): # For Python 3.8+ (ast.Num is deprecated)
        if isinstance(node.value, (int, float)):
            return node.value
        else:
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.Num): # For older Python versions (deprecated in 3.8)
        return node.n
    elif isinstance(node, ast.Name):
        if node.id in var_dict:
            return var_dict[node.id]
        else:
            raise ValueError(f"Undefined variable: {node.id}")
    elif isinstance(node, ast.BinOp):
        left = _eval_expr(node.left, var_dict)
        right = _eval_expr(node.right, var_dict)
        op_type = type(node.op)
        if op_type in _OPERATORS:
            return _OPERATORS[op_type](left, right)
        else:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_expr(node.operand, var_dict)
        op_type = type(node.op)
        if op_type in _OPERATORS:
            return _OPERATORS[op_type](operand)
        else:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
    elif isinstance(node, ast.Expression): # The root node of ast.parse(..., mode='eval')
        return _eval_expr(node.body, var_dict)
    else:
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")

def execute_calculation_string(calc_expr: str, var_dict: dict) -> str:
    """
    Computes and simplifies a mathematical expression string with variable substitution.

    Args:
        calc_expr (str): A string representing a mathematical expression.
        var_dict (dict): A dictionary where keys are variable names and
                         values are their corresponding numbers.

    Returns:
        str: The result of the simplified expression as a string.

    Raises:
        ValueError: If the input expression is invalid, contains unsupported
                    operations, or the computation fails.
    """
    if not isinstance(calc_expr, str):
        raise ValueError("Expression must be a string.")
    if not isinstance(var_dict, dict):
        raise ValueError("Variable dictionary must be a dict.")

    for var_name, var_value in var_dict.items():
        if not isinstance(var_name, str):
            raise ValueError(f"Variable name {var_name} must be a string.")
        if not isinstance(var_value, (int, float)):
            raise ValueError(f"Variable value for '{var_name}' must be a number (int or float).")

    try:
        # Using mode='eval' as we expect an expression, not a full module/statement
        node = ast.parse(calc_expr, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}")
    except Exception as e: # Catch other potential ast.parse errors
        raise ValueError(f"Error parsing expression: {e}")

    try:
        result = _eval_expr(node, var_dict)
        # Ensure the result is a number before converting to string
        if not isinstance(result, (int, float)):
             # This case should ideally be caught by _eval_expr's type checks
            raise ValueError(f"Evaluation resulted in non-numeric type: {type(result)}")
        return str(result)
    except ValueError: # Re-raise ValueErrors from _eval_expr
        raise
    except ZeroDivisionError:
        raise ValueError("Division by zero.")
    except Exception as e:
        raise ValueError(f"Error during expression evaluation: {e}")
