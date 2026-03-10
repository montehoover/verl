"""
A simple calculator module for parsing and evaluating mathematical expressions.

This module provides functionality to safely parse and calculate arithmetic
expressions provided as strings. It includes validation against unsupported
characters and potentially malicious constructs using Abstract Syntax Trees (AST).
"""
import re
import ast
import logging

# Regex to define allowed characters in an expression:
# ^                  matches the beginning of the string
# [0-9\s\+\-\*\/\(\)\.]*  matches zero or more occurrences of:
#   0-9              digits
#   \s               whitespace characters
#   \+               plus sign
#   \-               minus sign
#   \*               asterisk (multiplication)
#   \/               slash (division)
#   \(               opening parenthesis
#   \)               closing parenthesis
#   \.               dot (for decimal numbers)
# $                  matches the end of the string
ALLOWED_INPUT_PATTERN = re.compile(r"^[0-9\s\+\-\*\/\(\)\.]*$")

# Set of allowed AST node types. This is a security measure to restrict
# the kind of operations that can be evaluated, preventing execution of arbitrary code.
# For example, ast.Call and ast.Name are included here, but their usage is further
# restricted in `_is_node_allowed` to prevent calling arbitrary functions or using variables.
ALLOWED_AST_NODE_TYPES = {
    ast.Expression,  # Represents an expression.
    ast.Num,         # Represents a number (deprecated in Python 3.8+, use ast.Constant).
                     # For broader compatibility, we keep ast.Num. If ast.Constant is needed:
                     # ast.Constant, (handle str, bytes, bool, None, Ellipsis if necessary)
    ast.BinOp,       # Represents a binary operation (e.g., a + b).
    ast.UnaryOp,     # Represents a unary operation (e.g., -a).
    ast.USub,        # Represents unary subtraction.
    ast.UAdd,        # Represents unary addition.
    ast.Add,         # Represents addition.
    ast.Sub,         # Represents subtraction.
    ast.Mult,        # Represents multiplication.
    ast.Div,         # Represents division.
    ast.Call,        # Represents a function call (restricted in _is_node_allowed).
    ast.Name,        # Represents a variable or function name (restricted in _is_node_allowed).
    ast.Load         # Represents loading a variable's value.
}

# --- Logger Setup ---
# Create a logger instance
calculator_logger = logging.getLogger(__name__)
calculator_logger.setLevel(logging.INFO) # Set the logging level

# Create a file handler to write logs to a file
# Logs will be stored in 'calculator.log' in the current working directory
file_handler = logging.FileHandler('calculator.log')
file_handler.setLevel(logging.INFO)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
# Check if handlers are already added to prevent duplication if script is re-run in some environments
if not calculator_logger.handlers:
    calculator_logger.addHandler(file_handler)
# --- End Logger Setup ---


def _is_node_allowed(node: ast.AST) -> bool:
    """
    Recursively check if all nodes in the Abstract Syntax Tree (AST) are allowed.

    This function traverses the AST and verifies that each node's type is
    present in the `ALLOWED_AST_NODE_TYPES` set. It also explicitly disallows
    `ast.Call` (function calls) and `ast.Name` (variables/named constants)
    nodes to prevent execution of arbitrary functions or access to undefined variables
    in this simple calculator context.

    Args:
        node: The current AST node to check.

    Returns:
        True if the node and all its children are allowed, False otherwise.
    """
    if not isinstance(node, tuple(ALLOWED_AST_NODE_TYPES)):
        return False

    # Explicitly disallow function calls (ast.Call) and variable names (ast.Name)
    # for this simple calculator to prevent potential security risks.
    # If specific functions (e.g., 'abs') were to be allowed, this logic would need
    # to be more sophisticated, checking node.func.id for ast.Call, for example.
    if isinstance(node, (ast.Call, ast.Name)):
        return False

    for child_node in ast.iter_child_nodes(node):
        if not _is_node_allowed(child_node):
            return False
    return True


def parse_and_calculate(expression: str) -> float | int:
    """
    Parse and calculate a mathematical expression string.

    The function first validates the input string for allowed characters.
    Then, it parses the expression into an Abstract Syntax Tree (AST),
    validates the AST nodes to ensure no malicious or unsupported operations
    are present, compiles the AST, and finally evaluates it.

    Args:
        expression: The arithmetic expression string (e.g., "1 + 2 * 3").

    Returns:
        The numerical result of the calculation (either int or float).

    Raises:
        ValueError: If the expression string is invalid, contains unsupported
                    characters or operations, is empty, or if any calculation
                    error occurs (e.g., division by zero, syntax error).
    """
    calculator_logger.info(f"Attempting to parse and calculate expression: '{expression}'")

    if not isinstance(expression, str):
        err_msg = "Expression must be a string."
        calculator_logger.error(f"Validation failed for expression '{expression}': {err_msg}")
        raise ValueError(err_msg)

    # 1. Validate allowed characters in the raw expression string.
    # This is a preliminary check before more complex parsing.
    if not ALLOWED_INPUT_PATTERN.fullmatch(expression):
        err_msg = (
            "Expression contains unsupported characters. "
            "Only numbers, operators (+, -, *, /), parentheses, and whitespace are allowed."
        )
        calculator_logger.error(f"Validation failed for expression '{expression}': {err_msg}")
        raise ValueError(err_msg)

    # Ensure the expression is not empty or just whitespace.
    if not expression.strip():
        err_msg = "Expression cannot be empty."
        calculator_logger.error(f"Validation failed for expression '{expression}': {err_msg}")
        raise ValueError(err_msg)

    try:
        # 2. Parse the expression string into an Abstract Syntax Tree (AST).
        # `mode='eval'` is used because we expect a single expression that returns a value.
        parsed_expression_ast = ast.parse(expression, mode='eval')

        # 3. Validate all nodes in the AST (security measure).
        # This ensures that only allowed operations and constructs are present.
        if not _is_node_allowed(parsed_expression_ast):
            err_msg = "Expression contains disallowed operations or constructs."
            calculator_logger.error(f"AST validation failed for expression '{expression}': {err_msg}")
            raise ValueError(err_msg)

        # 4. Compile the AST into a code object.
        # The filename '<expression>' is used for error reporting.
        compiled_code_object = compile(parsed_expression_ast, filename='<expression>', mode='eval')

        # 5. Evaluate the compiled code object in a restricted environment.
        # `globals` and `locals` are restricted to prevent access to built-in functions
        # or variables beyond what's defined in the expression itself.
        # `{"__builtins__": {}}` creates an empty scope for builtins.
        result = eval(compiled_code_object, {"__builtins__": {}}, {})
        
        # Ensure the result of the evaluation is a number.
        if not isinstance(result, (int, float)):
            # This case should ideally not be reached if AST validation is robust
            # and only numeric operations are allowed.
            err_msg = "Calculation did not result in a number."
            calculator_logger.error(f"Calculation error for expression '{expression}': {err_msg}")
            raise ValueError(err_msg)
        
        calculator_logger.info(f"Successfully calculated expression '{expression}'. Result: {result}")
        return result

    except SyntaxError as e:
        err_msg = f"Invalid syntax in expression: {e}"
        calculator_logger.error(f"Syntax error for expression '{expression}': {err_msg}")
        raise ValueError(err_msg)
    except ZeroDivisionError:
        err_msg = "Division by zero is not allowed."
        calculator_logger.error(f"Calculation error for expression '{expression}': {err_msg}")
        raise ValueError(err_msg)
    except ValueError as e: # Catch specific ValueErrors raised above first
        calculator_logger.error(f"ValueError during processing of '{expression}': {e}")
        raise # Re-raise the already logged ValueError
    except Exception as e:
        # Catch any other unexpected errors during parsing or evaluation.
        err_msg = f"Could not compute expression: {e}"
        calculator_logger.error(f"Unexpected error for expression '{expression}': {err_msg}")
        raise ValueError(err_msg)

if __name__ == '__main__':
    # Example Usage:
    test_expressions = [
        ("1 + 1", 2),
        ("10 - 2 * 3", 4),
        ("(10 - 2) * 3", 24),
        ("10 / 2", 5.0),
        ("10 / 0", "ValueError"),
        ("1 + ", "ValueError"),
        ("1 + 'abc'", "ValueError"),
        ("import os", "ValueError"),
        ("print('hello')", "ValueError"),
        ("__import__('os').system('ls')", "ValueError"),
        ("2.5 * 4", 10.0),
        ("-5 + 2", -3),
        ("abs(-1)", "ValueError"), # abs is a Name node, disallowed
        ("1 + unknown_var", "ValueError") # unknown_var is a Name node, disallowed
    ]

    for expr, expected in test_expressions:
        try:
            result = parse_and_calculate(expr)
            print(f"Expression: '{expr}', Result: {result}, Expected: {expected}")
            if isinstance(expected, str) and expected == "ValueError":
                print("  FAILED: Expected ValueError but got result.")
            elif result != expected:
                print(f"  FAILED: Expected {expected} but got {result}.")
        except ValueError as e:
            print(f"Expression: '{expr}', Error: {e}, Expected: {expected}")
            if expected != "ValueError":
                print(f"  FAILED: Expected {expected} but got ValueError.")
        except Exception as e:
            print(f"Expression: '{expr}', UNEXPECTED Error: {e}")
