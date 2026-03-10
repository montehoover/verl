import re
import ast
import logging

# --- Logger Setup ---
# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler
file_handler = logging.FileHandler('calculator.log')
file_handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
# --- End Logger Setup ---

# Allowed characters: digits, operators (+, -, *, /), parentheses, and whitespace.
ALLOWED_CHARACTERS = re.compile(r"^[0-9\s\.\+\-\*\/\(\)]*$")

# Allowed AST node types
ALLOWED_NODES = {
    ast.Expression,   # The root node
    ast.Num,          # Numbers (deprecated in Python 3.8, use ast.Constant)
    ast.Constant,     # Numbers, strings, True, False, None
    ast.BinOp,        # Binary operations like +, -, *, /
    ast.UnaryOp,      # Unary operations like - (negation)
    ast.USub,         # Unary subtraction (negation)
    ast.UAdd,         # Unary addition (no-op)
    ast.Add,          # Addition operator
    ast.Sub,          # Subtraction operator
    ast.Mult,         # Multiplication operator
    ast.Div,          # Division operator
    # For more complex expressions, you might need to add:
    # ast.Pow,        # Power operator (**)
    # ast.Call,       # Function calls (if you want to allow safe functions like abs())
    # ast.Name,       # Names (variables, functions - be careful with this)
}

def _is_safe_node(node):
    """
    Recursively checks if all nodes in the AST are allowed.
    """
    if type(node) not in ALLOWED_NODES:
        return False
    for child_node in ast.iter_child_nodes(node):
        if not _is_safe_node(child_node):
            return False
    return True

def evaluate_expression(expr: str):
    """
    Parses a user-provided mathematical expression string and computes the result.

    Args:
        expr: str, a string representing the arithmetic expression.

    Returns:
        The computed result of the expression.

    Raises:
        ValueError: If unsupported characters, unsafe commands,
                    or invalid operations are detected in the input.
    """
    """
    Parses a user-provided mathematical expression string and computes the result.
    Logs the input expression and its outcome.

    Args:
        expr: str, a string representing the arithmetic expression.

    Returns:
        The computed result of the expression.

    Raises:
        TypeError: If the input expression is not a string.
        ValueError: If the expression is empty, contains unsupported characters,
                    has invalid syntax, contains unsafe operations, results in
                    division by zero, or encounters other evaluation errors.
    """
    logger.info(f"Received expression: '{expr}'")

    # --- Input Validation ---
    # Docstring: Check if the input expression is a string.
    if not isinstance(expr, str):
        logger.error(f"Type error for expression '{expr}': Not a string.")
        raise TypeError("Expression must be a string.")

    # Docstring: Check if the expression is empty or contains only whitespace.
    stripped_expr = expr.strip()
    if not stripped_expr:
        logger.error(f"Validation error for expression '{expr}': Empty or whitespace.")
        raise ValueError("Expression cannot be empty or just whitespace.")

    # Docstring: Check for unsupported characters in the expression.
    if not ALLOWED_CHARACTERS.match(stripped_expr):
        logger.error(f"Validation error for expression '{expr}': Unsupported characters.")
        raise ValueError(
            "Expression contains unsupported characters. "
            "Only numbers, operators (+, -, *, /), parentheses, and whitespace are allowed."
        )

    # --- AST Parsing and Validation ---
    node = None # Initialize node to None
    try:
        # Docstring: Parse the expression string into an Abstract Syntax Tree (AST).
        # The mode 'eval' is used as we expect a single expression.
        node = ast.parse(stripped_expr, mode='eval')
    except SyntaxError:
        # Docstring: Handle syntax errors during parsing.
        logger.error(f"Syntax error parsing expression '{expr}'.")
        raise ValueError("Invalid syntax in expression.")
    except Exception as e:
        # Docstring: Handle other potential errors during AST parsing.
        logger.error(f"Error parsing expression '{expr}': {e}")
        raise ValueError(f"Error parsing expression: {e}")

    # Docstring: Validate the AST to ensure it only contains allowed node types,
    # preventing execution of unsafe or unsupported operations.
    if not _is_safe_node(node):
        logger.error(f"AST validation failed for expression '{expr}': Unsafe/unsupported operations.")
        raise ValueError("Expression contains unsafe or unsupported operations.")

    # --- Expression Compilation and Evaluation ---
    try:
        # Docstring: Compile the validated AST node into a code object.
        # The '<string>' filename is a convention for code compiled from a string.
        code = compile(node, filename='<string>', mode='eval')
        
        # Docstring: Evaluate the compiled code object.
        # An empty dictionary for globals and locals restricts the execution environment,
        # enhancing safety by preventing access to built-in functions or global variables.
        result = eval(code, {"__builtins__": {}}, {})
        logger.info(f"Expression '{expr}' evaluated to: {result}")
        return result
    except ZeroDivisionError:
        # Docstring: Handle division by zero errors specifically.
        logger.error(f"Evaluation error for expression '{expr}': Division by zero.")
        raise ValueError("Division by zero is not allowed.")
    except Exception as e:
        # Docstring: Handle any other errors that occur during the evaluation phase.
        logger.error(f"Error evaluating expression '{expr}': {e}")
        raise ValueError(f"Error evaluating expression: {e}")

if __name__ == '__main__':
    # Example Usage:
    test_expressions = [
        "1 + 1",
        "10 - 5.5",
        "2 * 3",
        "10 / 2",
        "(1 + 2) * 3",
        "-5 + 2",
        "-(1 + 2)",
        "2.5 * 4 - 3 / 1.5",
        "  100  ",
        # Invalid expressions
        "1 + ",
        "1 + 1)",
        "1 +* 2",
        "import os",
        "__import__('os').system('echo unsafe')",
        "print('hello')",
        "a = 5",
        "1/0",
        "eval('1+1')",
        "some_function()"
    ]

    for exp in test_expressions:
        try:
            result = evaluate_expression(exp)
            print(f"'{exp}' => {result}")
        except (ValueError, TypeError) as e:
            print(f"'{exp}' => Error: {e}")

    print("\nTesting empty or whitespace expression:")
    try:
        evaluate_expression("   ")
    except ValueError as e:
        print(f"'   ' => Error: {e}")
    
    print("\nTesting expression with invalid characters:")
    try:
        evaluate_expression("1 + 1a")
    except ValueError as e:
        print(f"'1 + 1a' => Error: {e}")

    print("\nTesting expression with unsafe function call attempt (should be caught by AST check):")
    # This specific case might be caught by ALLOWED_CHARACTERS first if it has non-allowed chars,
    # but if it were "abs(5)" and abs was not in ALLOWED_NODES (and Call was), it would be caught.
    # For now, the char check is quite strict.
    # Let's try something that passes char check but not AST node check if we were more permissive with chars.
    # However, current ALLOWED_CHARACTERS is very strict.
    # If we allowed letters for functions, then an AST check for Call node would be critical.
    # For now, this will be caught by character validation.
    try:
        evaluate_expression("pow(2,3)") # 'pow' has letters, will be caught by ALLOWED_CHARACTERS
    except ValueError as e:
        print(f"'pow(2,3)' => Error: {e}")

    # Test for ast.Name (e.g. variables) - should be disallowed by current ALLOWED_NODES
    # This will be caught by ALLOWED_CHARACTERS if 'x' is not allowed.
    # If 'x' was allowed, ast.Name would need to be explicitly disallowed or handled.
    # Current ALLOWED_CHARACTERS prevents 'x'.
    # If we had `ALLOWED_CHARACTERS = re.compile(r"^[0-9\s\.\+\-\*\/\(\)a-zA-Z]*$")`
    # then the AST check for `ast.Name` would be crucial.
    # For now, this is fine.
    try:
        evaluate_expression("x + 1")
    except ValueError as e:
        print(f"'x + 1' => Error: {e}")
