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

    def _eval_node(node):
        if isinstance(node, ast.Num):  # <number>
            return node.n
        elif isinstance(node, ast.Name):  # <variable>
            if node.id in vars_mapping:
                return vars_mapping[node.id]
            else:
                raise ValueError(f"Undefined variable: {node.id}")
        elif isinstance(node, ast.BinOp):  # <left_operand> <operator> <right_operand>
            left_val = _eval_node(node.left)
            right_val = _eval_node(node.right)
            op_type = type(node.op)
            if op_type in _OPERATORS:
                return _OPERATORS[op_type](left_val, right_val)
            else:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub): # <operator> <operand> e.g., -5
            return -_eval_node(node.operand)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd): # <operator> <operand> e.g., +5
            return +_eval_node(node.operand)
        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    try:
        # Parse the formula string into an AST
        # mode='eval' ensures it's an expression
        node = ast.parse(formula_str.strip(), mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid formula syntax: {e}") from e
    except Exception as e:
        raise ValueError(f"Error parsing formula: {e}") from e
        
    if not isinstance(node, ast.Expression) or not node.body:
        raise ValueError("Invalid expression structure.")

    try:
        result = _eval_node(node.body)
        # Ensure result is a number before converting to string
        if not isinstance(result, (int, float)):
            raise ValueError("Expression did not evaluate to a numeric result.")
        return str(result)
    except ValueError: # Re-raise ValueErrors from _eval_node
        raise
    except ZeroDivisionError:
        raise ValueError("Division by zero.")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {e}") from e

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
