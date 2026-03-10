import re
import operator
import ast

# Helper dictionaries to map AST nodes to Python operators
_ARITHMETIC_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,  # For exponentiation (e.g., x ** y)
}

_UNARY_OPERATORS = {
    ast.USub: operator.neg,  # For negation (e.g., -x)
    ast.UAdd: operator.pos,   # For unary plus (e.g., +x)
}

def _evaluate_ast_node(node, variables_mapping: dict):
    """
    Recursively evaluates an AST node representing a part of a mathematical expression.

    Args:
        node: The AST node to evaluate.
        variables_mapping (dict): A mapping of variable names to their numeric values.

    Returns:
        The numeric result of evaluating the node.

    Raises:
        ValueError: If an unsupported node type, operator, undefined variable,
                    or non-numeric variable value is encountered.
    """
    if isinstance(node, ast.Constant):  # Handles numbers, None, True, False
        if isinstance(node.value, (int, float)):
            return node.value
        # Disallow other constant types like strings within the expression itself
        raise ValueError(f"Unsupported constant type in expression: {type(node.value).__name__}")
    elif isinstance(node, ast.Num):  # For Python < 3.8 compatibility (numbers)
        return node.n
    elif isinstance(node, ast.Name):  # Variable lookup
        variable_name = node.id
        if variable_name in variables_mapping:
            value = variables_mapping[variable_name]
            # Ensure the value from the mapping is numeric
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Variable '{variable_name}' has a non-numeric value: {value!r}"
                )
            return value
        else:
            raise ValueError(f"Undefined variable: {variable_name}")
    elif isinstance(node, ast.BinOp):  # Binary operations (e.g., a + b)
        left_operand_value = _evaluate_ast_node(node.left, variables_mapping)
        right_operand_value = _evaluate_ast_node(node.right, variables_mapping)
        operator_type = type(node.op)
        if operator_type in _ARITHMETIC_OPERATORS:
            return _ARITHMETIC_OPERATORS[operator_type](left_operand_value, right_operand_value)
        else:
            raise ValueError(f"Unsupported binary operator: {operator_type.__name__}")
    elif isinstance(node, ast.UnaryOp):  # Unary operations (e.g., -a)
        operand_value = _evaluate_ast_node(node.operand, variables_mapping)
        operator_type = type(node.op)
        if operator_type in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[operator_type](operand_value)
        else:
            raise ValueError(f"Unsupported unary operator: {operator_type.__name__}")
    else:
        # Catch any other unsupported AST node types
        raise ValueError(f"Unsupported expression component: {type(node).__name__}")


def simplify_math_expression(formula_str: str, vars_mapping: dict) -> str:
    """
    Processes a mathematical expression string, substitutes variables, and returns the result.

    Args:
        formula_str (str): A string representing a mathematical formula
                           potentially containing variables.
        vars_mapping (dict): A mapping of variable names to their numeric values
                             for evaluation.

    Returns:
        str: The result after computing the expression, returned in string format.

    Raises:
        TypeError: If formula_str is not a string or vars_mapping is not a dict.
        ValueError: If an error occurs due to an invalid expression,
                    undefined variables, non-numeric values where numbers are expected,
                    division by zero, or other unsuccessful processing.
    """
    if not isinstance(formula_str, str):
        raise TypeError("Formula string must be a string.")
    if not isinstance(vars_mapping, dict):
        raise TypeError("Variables mapping must be a dictionary.")

    # Validate variable names and their mapped values in vars_mapping upfront
    for var_name, value in vars_mapping.items():
        if not isinstance(var_name, str):
            raise ValueError(f"Variable name {var_name!r} must be a string.")
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Variable '{var_name}' is mapped to a non-numeric value: {value!r}"
            )

    try:
        # Parse the formula string into an Abstract Syntax Tree (AST).
        # 'eval' mode is used because we expect a single expression.
        parsed_ast_tree = ast.parse(formula_str, mode='eval')

        # The actual expression node is in parsed_ast_tree.body.
        # Evaluate the AST using the helper function.
        result = _evaluate_ast_node(parsed_ast_tree.body, vars_mapping)

        # Ensure the final result is a number before converting to string.
        # This should typically be guaranteed by _evaluate_ast_node's logic.
        if not isinstance(result, (int, float)):
            raise ValueError("Expression did not evaluate to a numeric result.")

        return str(result)

    except SyntaxError as e:
        # Raised by ast.parse for malformed expressions (e.g., "1 +", "a b c").
        raise ValueError(f"Invalid mathematical expression syntax: {e}")
    except ZeroDivisionError:
        # Specifically catch division by zero during evaluation.
        raise ValueError("Mathematical error: Division by zero.")
    # ValueErrors raised by _evaluate_ast_node or initial checks will propagate.
    # Catch any other unexpected errors during parsing or evaluation.
    except Exception as e:
        # Avoid leaking arbitrary internal exception types; wrap them in ValueError.
        # This check ensures we don't re-wrap ValueErrors already raised by our logic.
        if isinstance(e, ValueError) or isinstance(e, TypeError):
            raise
        raise ValueError(f"An unexpected error occurred while processing the expression: {e}")
