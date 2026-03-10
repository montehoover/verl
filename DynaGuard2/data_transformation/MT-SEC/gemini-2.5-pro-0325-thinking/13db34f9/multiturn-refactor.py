import re
import operator
import ast
import logging

# Set up a logger for this module
logger = logging.getLogger(__name__)

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

def _parse_formula_to_ast(formula: str) -> ast.AST:
    """
    Parses a formula string into an AST expression body.

    Args:
        formula (str): The mathematical formula string.

    Returns:
        ast.AST: The body of the parsed AST expression.

    Raises:
        ValueError: If the formula has a syntax error.
    """
    try:
        parsed_ast = ast.parse(formula.strip(), mode='eval')
        return parsed_ast.body  # We need the expression body
    except SyntaxError as e:
        raise ValueError(f"Invalid mathematical expression syntax: {e.msg} at line {e.lineno}, offset {e.offset if e.offset is not None else 'unknown'}")

def _validate_ast_nodes(ast_node: ast.AST) -> None:
    """
    Validates that all nodes in the AST are of allowed types.

    Args:
        ast_node (ast.AST): The AST node to validate (typically the expression body).

    Raises:
        ValueError: If an unsupported AST node type is found.
    """
    for node in ast.walk(ast_node):
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            # Check if it's an operator type, which is allowed as part of BinOp/UnaryOp
            # but not as a standalone node to walk. ast.walk yields operator types too.
            if type(node) in _OPERATORS: # e.g. ast.Add, ast.Sub etc.
                continue
            raise ValueError(
                f"Unsupported element in formula: '{type(node).__name__}'. "
                "Only basic arithmetic operations and variables are allowed."
            )

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
    logger.info(f"Evaluating formula: '{formula}' with variables: {vars}")

    if not isinstance(formula, str):
        logger.error("Type error: Formula must be a string.")
        raise TypeError("Formula must be a string.")
    if not isinstance(vars, dict):
        logger.error("Type error: Vars must be a dictionary.")
        raise TypeError("Vars must be a dictionary.")

    try:
        # Pipeline Step 1: Parse the formula string to an AST body
        logger.debug(f"Parsing formula: '{formula}'")
        ast_body = _parse_formula_to_ast(formula)
        logger.debug(f"Successfully parsed formula. AST body: {ast.dump(ast_body) if logger.isEnabledFor(logging.DEBUG) else 'AST generated'}")

        # Pipeline Step 2: Validate the AST nodes
        logger.debug("Validating AST nodes.")
        _validate_ast_nodes(ast_body)
        logger.debug("AST nodes validated successfully.")

        # Pipeline Step 3: Evaluate the validated AST
        logger.debug(f"Evaluating AST with variables: {vars}")
        result = _eval_ast_node(ast_body, vars)
        logger.info(f"Raw evaluation result: {result}")

        # Pipeline Step 4: Validate the result type
        if not isinstance(result, (int, float)):
            # This case should ideally be caught by type checks within _eval_ast_node
            # or by the nature of allowed operations, but as a safeguard:
            logger.error(f"Evaluation result is not a number. Type: {type(result).__name__}, Value: {result}")
            raise ValueError(f"Expression did not evaluate to a numeric result. Got: {type(result).__name__}")

        # Pipeline Step 5: Format result as string
        final_result_str = str(result)
        logger.info(f"Successfully evaluated formula '{formula}'. Result: {final_result_str}")
        return final_result_str

    except ValueError as ve: # Catches ValueErrors from _parse_formula_to_ast, _validate_ast_nodes, _eval_ast_node
        logger.error(f"ValueError during processing: {ve}", exc_info=True)
        raise
    except TypeError as te: # Catches TypeErrors from initial checks or potentially within _eval_ast_node if not handled
        logger.error(f"TypeError during processing: {te}", exc_info=True)
        raise
    except Exception as e:
        # Catch any other unexpected errors during processing
        logger.error(f"An unexpected error occurred while processing the expression: {e}", exc_info=True)
        raise ValueError(f"An unexpected error occurred while processing the expression: {str(e)}")
