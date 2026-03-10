import re
import operator
import ast

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

def simplify_math_expression(formula_str: str, vars_mapping: dict) -> str:
    """
    Computes and simplifies a mathematical formula string with variables,
    returning the result as a string.

    Parameters:
    formula_str (str): The mathematical formula with variables.
    vars_mapping (dict): A dictionary mapping variable names to their numeric values.

    Returns:
    str: The calculated result as a string.

    Raises:
    ValueError: If the expression is invalid, a variable is undefined,
                a variable has a non-numeric value, or for other processing issues.
    """
    if not isinstance(vars_mapping, dict):
        raise ValueError("vars_mapping must be a dictionary.")

    try:
        # Parse the expression to an Abstract Syntax Tree (AST)
        node = ast.parse(formula_str, mode='eval')

        def _eval_node(node, current_vars):
            if isinstance(node, ast.Num): # Python 3.7 and earlier
                return node.n
            if isinstance(node, ast.Constant): # Python 3.8 and later
                return node.value
            elif isinstance(node, ast.Name):
                var_name = node.id
                if var_name in current_vars:
                    val = current_vars[var_name]
                    if not isinstance(val, (int, float)):
                        raise ValueError(f"Variable '{var_name}' has non-numeric value: {val}")
                    return val
                else:
                    raise ValueError(f"Undefined variable: {var_name}")
            elif isinstance(node, ast.BinOp):
                left_val = _eval_node(node.left, current_vars)
                right_val = _eval_node(node.right, current_vars)
                op_type = type(node.op)
                if op_type in _ALLOWED_OPS:
                    if op_type == ast.Div and right_val == 0:
                        raise ZeroDivisionError("division by zero")
                    return _ALLOWED_OPS[op_type](left_val, right_val)
                else:
                    raise ValueError(f"Unsupported operator: {op_type.__name__}")
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub): # Handles negative numbers
                operand_val = _eval_node(node.operand, current_vars)
                return -operand_val
            elif isinstance(node, ast.Expression): # The top level node
                return _eval_node(node.body, current_vars)
            else:
                raise ValueError(f"Unsupported expression structure: {type(node).__name__}")

        result = _eval_node(node, vars_mapping)
        
        # Ensure the result is a number (int or float) before converting to string
        if not isinstance(result, (int, float)):
            raise ValueError("Expression did not evaluate to a number")
            
        return str(result)

    except (SyntaxError, TypeError, ZeroDivisionError) as e:
        raise ValueError(f"Invalid or malformed formula: {e}") from e
    except ValueError: # Re-raise ValueErrors from _eval_node or initial checks
        raise
    except Exception as e: # Catch any other unexpected errors during parsing or evaluation
        raise ValueError(f"Could not process formula: {e}") from e
