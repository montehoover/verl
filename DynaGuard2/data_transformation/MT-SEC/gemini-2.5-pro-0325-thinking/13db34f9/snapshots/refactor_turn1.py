import re
import operator
import ast

# Allowed AST node types for safety.
# This set defines what kind of Python constructs are permitted in the formula.
# We are aiming for basic arithmetic.
_ALLOWED_NODE_TYPES = (
    ast.Expression,  # The root node for mode='eval'
    ast.Constant,    # For numbers (Python 3.8+) and potentially strings if allowed
    ast.Num,         # For numbers (Python < 3.8)
    ast.Name,        # For variables
    ast.Load,        # Context for loading a variable's value
    ast.BinOp,       # For binary operations like +, -, *, /
    ast.UnaryOp,     # For unary operations like - (negation)
    # Specific operator types (these are types of node.op, not nodes themselves)
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,        # Unary plus
)

# Map AST operator nodes to actual operator functions
_OPERATORS = {
    # Binary operators
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv, # Use truediv for float division
    ast.Pow: operator.pow,
    # Unary operators
    ast.USub: operator.neg,
    ast.UAdd: operator.pos, # Unary plus, usually a no-op for numbers
}

def _eval_ast_node(node, variables: dict):
    """
    Recursively evaluates a pre-validated AST node.
    Assumes all nodes in the AST rooted at `node` are of allowed types.
    """
    node_type = type(node)

    if node_type is ast.Constant: # Python 3.8+
        # We expect numeric values from variables, and numbers in the formula.
        return node.value
    elif node_type is ast.Num:  # For Python < 3.8 (numbers)
        return node.n
    elif node_type is ast.Name:
        var_name = node.id
        if var_name in variables:
            val = variables[var_name]
            # Ensure variables are numbers
            if not isinstance(val, (int, float)):
                raise ValueError(f"Variable '{var_name}' must be a number, got {type(val).__name__}.")
            return val
        else:
            raise ValueError(f"Undefined variable: '{var_name}'")
    elif node_type is ast.BinOp:
        left_val = _eval_ast_node(node.left, variables)
        right_val = _eval_ast_node(node.right, variables)
        op_func = _OPERATORS.get(type(node.op))
        if op_func:
            if not isinstance(left_val, (int, float)) or not isinstance(right_val, (int, float)):
                raise ValueError("Binary operations can only be performed on numbers.")
            try:
                return op_func(left_val, right_val)
            except ZeroDivisionError:
                raise ValueError("Division by zero.")
            except TypeError:
                raise ValueError(f"Type error during binary operation {type(node.op).__name__}.")
        else:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
    elif node_type is ast.UnaryOp:
        operand_val = _eval_ast_node(node.operand, variables)
        op_func = _OPERATORS.get(type(node.op))
        if op_func:
            if not isinstance(operand_val, (int, float)):
                raise ValueError("Unary operations can only be performed on numbers.")
            try:
                return op_func(operand_val)
            except TypeError:
                 raise ValueError(f"Type error during unary operation {type(node.op).__name__}.")
        else:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
    else:
        raise ValueError(f"Execution of unsupported AST node type: {node_type.__name__}")


def evaluate_math_expression(formula: str, vars: dict) -> str:
    """
    Processes a mathematical expression string, substitutes variables,
    and returns the calculated, simplified result as a string.

    Args:
        formula (str): A string representing a mathematical formula
                       potentially containing variables.
        vars (dict): A mapping of variable names to their numeric values
                     for evaluation. Numeric values are expected to be int or float.

    Returns:
        str: The result after computing the expression, returned in string format.

    Raises:
        ValueError: If an error occurs due to an invalid expression,
                    unsupported operations, undefined variables, or
                    unsuccessful processing.
    """
    if not isinstance(formula, str):
        raise TypeError("Formula must be a string.")
    if not isinstance(vars, dict):
        raise TypeError("Vars must be a dictionary.")

    try:
        parsed_ast = ast.parse(formula.strip(), mode='eval')

        for node in ast.walk(parsed_ast):
            if not isinstance(node, _ALLOWED_NODE_TYPES):
                raise ValueError(
                    f"Unsupported element in formula: '{type(node).__name__}'. "
                    "Only basic arithmetic operations and variables are allowed."
                )

        result = _eval_ast_node(parsed_ast.body, vars)

        if not isinstance(result, (int, float)):
            raise ValueError(f"Expression did not evaluate to a number. Got: {type(result).__name__}")

        return str(result)

    except SyntaxError as e:
        raise ValueError(f"Invalid mathematical expression syntax: {e.msg} at line {e.lineno}, offset {e.offset if e.offset is not None else 'unknown'}")
    except ValueError: # Re-raise ValueErrors from _eval_ast_node or validation
        raise
    except Exception as e:
        raise ValueError(f"Error processing expression: {str(e)}")
