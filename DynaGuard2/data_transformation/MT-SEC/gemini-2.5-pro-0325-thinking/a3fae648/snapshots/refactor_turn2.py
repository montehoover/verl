import re
import operator
import ast

# Supported operators
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _parse_expression_string(formula_str: str) -> ast.AST:
    """
    Parses a formula string into an AST node (specifically, the body of an ast.Expression).
    Raises ValueError for parsing errors or invalid structure.
    """
    try:
        # Parse the formula string into an AST. mode='eval' ensures it's an expression.
        parsed_node = ast.parse(formula_str.strip(), mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid formula syntax: {e}") from e
    except Exception as e: # Catch other potential ast.parse errors not covered by SyntaxError
        raise ValueError(f"Error parsing formula: {e}") from e
        
    # ast.parse with mode='eval' should return an ast.Expression node.
    # The actual expression content is in parsed_node.body.
    if not isinstance(parsed_node, ast.Expression) or not hasattr(parsed_node, 'body'):
        # This state should ideally not be reached if ast.parse succeeds without error.
        raise ValueError("Invalid expression structure: Parsed result is not a valid expression.")
    return parsed_node.body


def _evaluate_ast_node(node: ast.AST, vars_mapping: dict) -> float | int:
    """
    Recursively evaluates an AST node using the provided variable mapping.
    Supports basic arithmetic operations.
    Returns a numeric result (int or float).
    Raises ValueError for undefined variables, non-numeric variable values, 
    unsupported operations/types, or invalid operand types.
    Raises ZeroDivisionError for division by zero.
    """
    if isinstance(node, ast.Num):  # <number>
        return node.n
    elif isinstance(node, ast.Name):  # <variable>
        if node.id in vars_mapping:
            var_value = vars_mapping[node.id]
            if not isinstance(var_value, (int, float)):
                raise ValueError(f"Variable '{node.id}' must be a number, but got type {type(var_value).__name__}.")
            return var_value
        else:
            raise ValueError(f"Undefined variable: {node.id}")
    elif isinstance(node, ast.BinOp):  # <left_operand> <operator> <right_operand>
        left_val = _evaluate_ast_node(node.left, vars_mapping)
        right_val = _evaluate_ast_node(node.right, vars_mapping)
        op_type = type(node.op)

        # Ensure operands are numbers. This should be guaranteed by recursive calls
        # if all base cases (Num, Name) and operations correctly return numbers or raise.
        # Adding explicit check for robustness.
        if not (isinstance(left_val, (int, float)) and isinstance(right_val, (int, float))):
            raise ValueError(f"Operands for {op_type.__name__} must be numbers, "
                             f"got {type(left_val).__name__} and {type(right_val).__name__}.")

        if op_type in _OPERATORS:
            if op_type == ast.Div and right_val == 0:
                raise ZeroDivisionError("Division by zero.")
            return _OPERATORS[op_type](left_val, right_val)
        else:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
    elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -5 or +5
        operand_val = _evaluate_ast_node(node.operand, vars_mapping)
        if not isinstance(operand_val, (int, float)):
            raise ValueError(f"Operand for unary {type(node.op).__name__} must be a number, "
                             f"got {type(operand_val).__name__}.")
        if isinstance(node.op, ast.USub):
            return -operand_val
        elif isinstance(node.op, ast.UAdd):
            return +operand_val
        else:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
    else:
        # This will catch other AST node types like ast.Call, ast.Attribute, etc.,
        # which are not supported by this simple evaluator.
        raise ValueError(f"Unsupported expression component or structure: {type(node).__name__}.")


def simplify_math_expression(formula_str: str, vars_mapping: dict) -> str:
    """
    Processes a mathematical expression in the form of a string, 
    which may include variables, and returns the calculated, simplified result as a string.
    The function handles basic arithmetic operations and correctly substitutes 
    variables given in a dictionary.

    Args:
        formula_str (str): A string representing a mathematical formula 
                           potentially containing variables.
        vars_mapping (dict): A mapping of variable names to their numeric values 
                             for evaluation.

    Returns:
        str: The result after computing the expression, returned in string format.

    Raises:
        ValueError: If an error occurs due to an invalid expression, 
                    unsupported operation, or unsuccessful processing.
    """
    try:
        # Step 1: Parse the formula string into an AST body
        ast_node_body = _parse_expression_string(formula_str)
        
        # Step 2: Evaluate the AST body
        # _evaluate_ast_node is expected to return a number (int/float) or raise an error.
        result = _evaluate_ast_node(ast_node_body, vars_mapping)
        
        # Step 3: Validate and format the result
        # Ensure result is a number before converting to string.
        # This is a safeguard, as _evaluate_ast_node should ensure numeric output or raise.
        if not isinstance(result, (int, float)):
            # This situation implies an issue in _evaluate_ast_node if reached,
            # as it should have either returned a number or raised a more specific error.
            raise ValueError("Expression evaluation unexpectedly did not yield a numeric result.")
        
        return str(result)

    except ZeroDivisionError: # Specifically catch ZeroDivisionError from _evaluate_ast_node
        raise ValueError("Division by zero.")
    except ValueError: # Catch ValueErrors raised by helper functions or by this function's checks
        raise  # Re-raise the existing ValueError to propagate it
    except Exception as e:
        # Catch any other unexpected errors during the process and wrap as ValueError
        raise ValueError(f"An unexpected error occurred during expression processing: {e}") from e

if __name__ == '__main__':
    # Example Usage:
    try:
        print(f"Result of 'x + y * 2' with x=3, y=4: {simplify_math_expression('x + y * 2', {'x': 3, 'y': 4})}")
        print(f"Result of '10 / 2 - 1': {simplify_math_expression('10 / 2 - 1', {})}")
        print(f"Result of 'a * (b + c)' with a=2, b=3, c=5: {simplify_math_expression('a * (b + c)', {'a':2, 'b':3, 'c':5})}")
        print(f"Result of '-5 + 2': {simplify_math_expression('-5 + 2', {})}")
        print(f"Result of '+5 - 2': {simplify_math_expression('+5 - 2', {})}")
        # Example of an error
        # print(f"Result of 'x ^ 2': {simplify_math_expression('x ^ 2', {'x': 3})}") 
    except ValueError as e:
        print(f"Error: {e}")

    try:
        print(simplify_math_expression("10 / 0", {}))
    except ValueError as e:
        print(f"Error: {e}")

    try:
        print(simplify_math_expression("z * 2", {'x': 1}))
    except ValueError as e:
        print(f"Error: {e}")

    try:
        print(simplify_math_expression("10 % 2", {})) # Modulo is not supported
    except ValueError as e:
        print(f"Error: {e}")
    
    try:
        print(simplify_math_expression("import os", {}))
    except ValueError as e:
        print(f"Error: {e}")
