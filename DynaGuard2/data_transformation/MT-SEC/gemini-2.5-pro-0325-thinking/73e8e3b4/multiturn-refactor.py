import re
import ast
import logging

# Configure basic logging
# This will log to the console by default.
# You can customize this further (e.g., log to a file, set different levels per module).
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Whitelist of allowed AST node types
# Defined globally as it's used by _parse_and_validate_expression
_ALLOWED_NODES = {
    ast.Expression,  # The top-level node for an expression
    ast.Num,         # Numbers (deprecated in Python 3.8, use ast.Constant)
    ast.Constant,    # Numbers, strings, None, True, False
    ast.BinOp,       # Binary operations like +, -, *, /
    ast.UnaryOp,     # Unary operations like - (negation)
    ast.Add,         # Addition operator
    ast.Sub,         # Subtraction operator
    ast.Mult,        # Multiplication operator
    ast.Div,         # Division operator
    ast.USub,        # Unary subtraction (negation)
    ast.UAdd,        # Unary addition (identity)
}

def _parse_and_validate_expression(calc_expr: str) -> ast.AST:
    """
    Parses the expression string into an AST and validates its nodes.

    Args:
        calc_expr: The mathematical expression string.

    Returns:
        A validated AST node.

    Raises:
        ValueError: If the expression is invalid or contains unsupported elements.
    """
    # Allow numbers, basic arithmetic operators, parentheses, and whitespace.
    # Disallow any letters or other symbols to prevent function calls or variable names.
    if not re.fullmatch(r"[0-9\s\.\+\-\*\/\(\)]+", calc_expr):
        err_msg = f"Validation failed for expression '{calc_expr}': Expression contains unsupported characters."
        logger.error(err_msg)
        raise ValueError("Expression contains unsupported characters.")

    try:
        # Parse the expression into an AST node
        node = ast.parse(calc_expr, mode='eval')
    except SyntaxError as e:
        err_msg = f"Syntax error in expression '{calc_expr}': {e}"
        logger.error(err_msg)
        raise ValueError(f"Invalid syntax in expression: {e}")
    except Exception as e:
        err_msg = f"Error parsing expression '{calc_expr}': {e}"
        logger.error(err_msg)
        raise ValueError(f"Error parsing expression: {e}")

    # Validate all nodes in the AST
    for sub_node in ast.walk(node):
        if not isinstance(sub_node, tuple(_ALLOWED_NODES)):
            # Special handling for ast.Num in older Python versions if ast.Constant is preferred
            if isinstance(sub_node, ast.Num) and ast.Constant in _ALLOWED_NODES:
                continue # Allow ast.Num if ast.Constant is in _ALLOWED_NODES for compatibility
            err_msg = f"Validation failed for expression '{calc_expr}': Unsupported operation or node type: {type(sub_node).__name__}"
            logger.error(err_msg)
            raise ValueError(f"Unsupported operation or node type: {type(sub_node).__name__}")
        
        # Ensure ast.Constant is a number (int or float)
        if isinstance(sub_node, ast.Constant) and not isinstance(sub_node.value, (int, float)):
            err_msg = f"Validation failed for expression '{calc_expr}': Only numeric constants are allowed. Found: {sub_node.value}"
            logger.error(err_msg)
            raise ValueError("Only numeric constants are allowed.")
    return node

def _evaluate_ast_node(node: ast.AST):
    """
    Evaluates a validated AST node.

    Args:
        node: The AST node to evaluate.

    Returns:
        The result of the evaluation.

    Raises:
        ValueError: If evaluation fails (e.g., division by zero).
    """
    try:
        # Compile the AST node into a code object
        # The mode 'eval' is used because we expect a single expression
        code = compile(node, filename='<string>', mode='eval')
        
        # Evaluate the compiled code object
        # Provide an empty dictionary for globals and locals to restrict context
        result = eval(code, {"__builtins__": {}}, {})
        return result
    except ZeroDivisionError as e:
        err_msg = f"Evaluation failed: Division by zero."
        logger.error(err_msg)
        raise ValueError("Division by zero is not allowed.")
    except Exception as e:
        # Catch any other errors during evaluation
        err_msg = f"Error evaluating AST node: {e}"
        logger.error(err_msg)
        raise ValueError(f"Error evaluating expression: {e}")

def evaluate_and_compute(calc_expr: str):
    """
    Evaluates a mathematical expression and returns the result.

    Args:
        calc_expr: str, a string containing a mathematical expression to evaluate.

    Returns:
        The evaluated result of the expression.

    Raises:
        ValueError: If the expression includes unsupported operators, characters,
                    unsafe code, or if evaluation fails.
    """
    logger.info(f"Attempting to evaluate expression: '{calc_expr}'")
    try:
        node = _parse_and_validate_expression(calc_expr)
        result = _evaluate_ast_node(node)
        logger.info(f"Successfully evaluated expression '{calc_expr}': Result = {result}")
        return result
    except ValueError as e:
        # Errors from helper functions are already logged, but we can log the top-level failure too.
        logger.error(f"Failed to evaluate expression '{calc_expr}': {e}")
        raise # Re-raise the caught ValueError to maintain original behavior

if __name__ == '__main__':
    # Example Usage and Basic Tests
    test_expressions = {
        "1 + 1": 2,
        "10 - 5": 5,
        "3 * 7": 21,
        "20 / 4": 5.0,
        "(1 + 2) * 3": 9,
        "10 / (2 + 3)": 2.0,
        "-5 + 10": 5,
        "-(5 + 5)": -10,
        "3.14 * 2": 6.28,
        "100 / 0.5": 200.0,
    }

    print("Running tests...")
    for expr, expected in test_expressions.items():
        try:
            result = evaluate_and_compute(expr)
            print(f"Expression: '{expr}', Result: {result}, Expected: {expected} -> {'PASS' if result == expected else 'FAIL'}")
        except ValueError as e:
            print(f"Expression: '{expr}', Error: {e} -> FAIL (unexpected error)")

    error_expressions = [
        "1 + a",  # unsupported character 'a'
        "import os", # unsupported operation
        "__import__('os').system('clear')", # unsafe code
        "1 / 0", # division by zero
        "1 +", # syntax error
        "eval('1+1')", # unsupported operation (Call)
        "print('hello')", # unsupported operation (Call)
        "abs(-5)", # unsupported operation (Call)
        "'hello' + 'world'", # non-numeric constant
    ]
    print("\nRunning error case tests...")
    for expr in error_expressions:
        try:
            evaluate_and_compute(expr)
            print(f"Expression: '{expr}' -> FAIL (expected ValueError)")
        except ValueError as e:
            print(f"Expression: '{expr}', Error: {e} -> PASS (ValueError raised as expected)")
        except Exception as e:
            print(f"Expression: '{expr}', Unexpected Error: {e} -> FAIL")
