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

def _evaluate_node(node, var_dict: dict):
    """
    Recursively evaluates an AST node.
    Helper for _compute_ast.
    """
    if isinstance(node, ast.Constant): # For Python 3.8+ (ast.Num is deprecated)
        if isinstance(node.value, (int, float)):
            return node.value
        else:
            # This case might not be reachable if ast.parse only creates Constants for supported types
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.Num): # For older Python versions (deprecated in 3.8)
        return node.n # ast.Num always holds a number
    elif isinstance(node, ast.Name):
        if node.id in var_dict:
            val = var_dict[node.id]
            # Ensure that the value substituted for a variable is numeric
            if not isinstance(val, (int, float)):
                raise ValueError(f"Variable '{node.id}' resolved to a non-numeric value of type {type(val).__name__}.")
            return val
        else:
            raise ValueError(f"Undefined variable: {node.id}")
    elif isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left, var_dict)
        right = _evaluate_node(node.right, var_dict)
        op_type = type(node.op)
        if op_type in _OPERATORS:
            # Specific check for division by zero before performing the operation
            if op_type is ast.Div and right == 0:
                raise ZeroDivisionError("Division by zero.")
            return _OPERATORS[op_type](left, right)
        else:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
    elif isinstance(node, ast.UnaryOp):
        operand = _evaluate_node(node.operand, var_dict)
        op_type = type(node.op)
        if op_type in _OPERATORS:
            return _OPERATORS[op_type](operand)
        else:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
    elif isinstance(node, ast.Expression): # The root node of ast.parse(..., mode='eval')
        return _evaluate_node(node.body, var_dict)
    else:
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")

# The following will be replaced by the new structure for execute_calculation_string and its helpers.
# This SEARCH block targets the original execute_calculation_string function.
# The REPLACE block will provide the new helper functions and the updated execute_calculation_string.
def _parse_expression_to_ast(calc_expr: str) -> ast.AST:
    """
    Parses a mathematical expression string into an AST node.

    Args:
        calc_expr (str): The mathematical expression string.

    Returns:
        ast.AST: The root AST node of the parsed expression.

    Raises:
        ValueError: If the expression syntax is invalid or parsing fails.
    """
    try:
        # Using mode='eval' as we expect an expression, not a full module/statement
        node = ast.parse(calc_expr, mode='eval')
        return node
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}")
    except Exception as e: # Catch other potential ast.parse errors
        raise ValueError(f"Error parsing expression: {e}")

def _compute_ast(node: ast.AST, var_dict: dict):
    """
    Computes the result of an AST node using the provided variable dictionary.

    Args:
        node (ast.AST): The AST node to evaluate.
        var_dict (dict): Dictionary of variable names to their numeric values.

    Returns:
        The numerical result of the computation (int or float).

    Raises:
        ValueError: If evaluation fails due to undefined variables, unsupported
                    operations, or non-numeric results.
        ZeroDivisionError: If division by zero occurs (propagated from _evaluate_node).
    """
    try:
        result = _evaluate_node(node, var_dict)
        # _evaluate_node is expected to return int/float or raise error.
        # This check is an additional safeguard.
        if not isinstance(result, (int, float)):
            raise ValueError(f"Internal error: Evaluation unexpectedly resulted in non-numeric type: {type(result).__name__}")
        return result
    except ZeroDivisionError: # Propagate to be handled by execute_calculation_string
        raise
    except ValueError: # Propagate ValueErrors from _evaluate_node
        raise
    except Exception as e: # Catch any other unexpected errors during evaluation
        # Wrap unexpected errors into ValueError for consistent error handling by the caller.
        raise ValueError(f"Unexpected error during expression computation: {e}")

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
            # Ensure variable names are strings
            raise ValueError(f"Variable name {repr(var_name)} must be a string.")
        if not isinstance(var_value, (int, float)):
            # Ensure variable values are numbers
            raise ValueError(f"Variable value for '{var_name}' must be a number (int or float), got {type(var_value).__name__}.")

    try:
        ast_node = _parse_expression_to_ast(calc_expr)
        # _compute_ast will use the validated var_dict
        result = _compute_ast(ast_node, var_dict)
        return str(result)
    except ZeroDivisionError: # Specifically catch and convert to ValueError as per requirements
        raise ValueError("Division by zero.")
    except ValueError: # ValueErrors from parsing or computation are already well-formed
        raise
    # Any other unexpected non-ValueError, non-ZeroDivisionError exceptions from helpers
    # would propagate if not caught by _compute_ast's generic Exception handler.
    # Since _compute_ast wraps generic exceptions, this path should mostly see ValueErrors.
