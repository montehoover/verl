import re
import ast
import logging

# --- Logger Setup ---
# Create a logger instance for this module.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set the minimum logging level. DEBUG for more verbosity.

# Create a file handler to write logs to a file.
# Logs will be stored in 'expression_evaluator.log' in the current working directory.
file_handler = logging.FileHandler("expression_evaluator.log")
file_handler.setLevel(logging.INFO)

# Create a formatter to define the log message format.
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger.
# This ensures that log messages are written to the specified file.
# To prevent duplicate handlers if this module is reloaded (e.g., in an interactive session),
# clear existing handlers first or check if handlers already exist.
if not logger.handlers:
    logger.addHandler(file_handler)

# --- Module-Level Constants for AST Validation ---
# These constants define the types of Abstract Syntax Tree (AST) nodes and operators
# that are permitted during the parsing and evaluation of mathematical expressions.
# This whitelist approach helps prevent the evaluation of unsafe or unsupported operations.

_ALLOWED_NODE_TYPES = (
    ast.Expression,  # Represents the overall expression.
    ast.Constant,    # Represents literal values like numbers (Python 3.8+). Also True, False, None.
    ast.Num,         # Represents numbers (deprecated in Python 3.8, superseded by ast.Constant).
    ast.BinOp,       # Represents binary operations (e.g., a + b, c * d).
    ast.UnaryOp,     # Represents unary operations (e.g., -a).
)

_ALLOWED_OPERATOR_TYPES = (
    ast.Add,         # The '+' addition operator.
    ast.Sub,         # The '-' subtraction operator.
    ast.Mult,        # The '*' multiplication operator.
    ast.Div,         # The '/' division operator.
    ast.USub,        # The unary '-' negation operator.
)

# Regular expression to validate the characters allowed in the input expression string.
# It permits:
#   \d      - digits (0-9)
#   \s      - whitespace characters (space, tab, newline, etc.)
#   \.      - a literal dot (for decimal points in floating-point numbers)
#   \+      - a literal plus sign
#   \-      - a literal minus sign
#   \*      - a literal asterisk (for multiplication)
#   \/      - a literal forward slash (for division)
#   \(      - a literal opening parenthesis
#   \)      - a literal closing parenthesis
# The `+` after the character set `[]` means one or more occurrences of these characters.
# `re.fullmatch` ensures the entire string consists only of these characters.
_VALID_EXPRESSION_PATTERN = re.compile(r"[\d\s\.\+\-\*\/\(\)]+")


# --- Helper Functions for Expression Evaluation ---

def _is_ast_safe(node: ast.AST) -> bool:
    """
    Recursively traverses an Abstract Syntax Tree (AST) to ensure all nodes
    and operators within it are of types explicitly allowed by the whitelist
    constants (_ALLOWED_NODE_TYPES, _ALLOWED_OPERATOR_TYPES).

    This function is crucial for security, preventing the evaluation of arbitrary
    Python code or unsupported mathematical operations.

    Args:
        node: The root ast.AST node of the tree (or subtree) to validate.

    Returns:
        True if all nodes and operators in the AST are allowed, False otherwise.
    """
    for ast_node in ast.walk(node):
        # Check if the node type itself is allowed.
        if not isinstance(ast_node, _ALLOWED_NODE_TYPES):
            return False  # Disallowed node type.

        # For binary operations (e.g., a + b), check if the operator (e.g., +) is allowed.
        if isinstance(ast_node, ast.BinOp) and not isinstance(ast_node.op, _ALLOWED_OPERATOR_TYPES):
            return False  # Disallowed binary operator.

        # For unary operations (e.g., -a), check if the operator (e.g., unary -) is allowed.
        if isinstance(ast_node, ast.UnaryOp) and not isinstance(ast_node.op, _ALLOWED_OPERATOR_TYPES):
            return False  # Disallowed unary operator.

        # For constant nodes (Python 3.8+), check the type of the constant's value.
        # We allow numbers (int, float). Booleans and None are also syntactically valid
        # ast.Constant values, but a later check ensures the final *result* is numeric.
        if isinstance(ast_node, ast.Constant):
            if not isinstance(ast_node.value, (int, float, bool, type(None))):
                return False  # Disallowed type for constant value (e.g., string, bytes).
        
        # For number nodes (ast.Num, deprecated but handled for compatibility),
        # ensure the number is an integer or float.
        if isinstance(ast_node, ast.Num):
            if not isinstance(ast_node.n, (int, float)):
                return False  # Disallowed type for ast.Num value (e.g., complex number).
                
    return True # All nodes and operators are safe.


def _validate_input_expression(math_expr: str) -> str:
    """
    Validates the raw input mathematical expression string.

    Checks for:
    1. Correct type (must be a string).
    2. Non-empty content after stripping whitespace.
    3. Allowed characters using a regular expression.

    Args:
        math_expr: The raw mathematical expression string.

    Returns:
        The stripped and validated mathematical expression string.

    Raises:
        TypeError: If `math_expr` is not a string.
        ValueError: If `math_expr` is empty, contains only whitespace,
                    or includes unsupported characters.
    """
    # Ensure the input is a string.
    if not isinstance(math_expr, str):
        logger.error(f"Type validation failed: Input expression is not a string (type: {type(math_expr)}).")
        raise TypeError("Input expression must be a string.")

    # Remove leading/trailing whitespace.
    stripped_expr = math_expr.strip()

    # Ensure the expression is not empty after stripping.
    if not stripped_expr:
        logger.error(f"Validation failed: Expression is empty or contains only whitespace. Original: '{math_expr}'")
        raise ValueError("Expression is empty or contains only whitespace.")

    # Validate the expression against the allowed character pattern.
    # re.fullmatch ensures the entire string conforms to the pattern.
    if not _VALID_EXPRESSION_PATTERN.fullmatch(stripped_expr):
        logger.error(f"Validation failed: Expression '{stripped_expr}' contains unsupported characters.")
        raise ValueError("Expression contains unsupported characters.")
    
    logger.info(f"Input expression '{stripped_expr}' passed character validation.")
    return stripped_expr


def _parse_and_validate_ast(expression_string: str) -> ast.AST:
    """
    Parses a validated mathematical expression string into an Abstract Syntax Tree (AST)
    and then validates this AST for safety.

    Args:
        expression_string: A validated (e.g., by `_validate_input_expression`)
                           mathematical expression string.

    Returns:
        A validated, safe ast.AST object representing the expression.

    Raises:
        ValueError: If the expression string has invalid syntax (e.g., unbalanced
                    parentheses) or if the resulting AST contains unsupported
                    operations or unsafe code constructs.
    """
    try:
        # Parse the expression string into an AST.
        # 'eval' mode is used because we expect a single expression that returns a value.
        logger.info(f"Parsing expression string: '{expression_string}'")
        ast_tree = ast.parse(expression_string, mode='eval')
    except SyntaxError as e:
        # This catches syntax errors like "1 + * 2" or "((1+2)".
        logger.error(f"Syntax error while parsing expression '{expression_string}': {e}")
        raise ValueError("Invalid syntax in expression.")

    # Perform a safety check on the generated AST.
    logger.info(f"Validating AST for expression: '{expression_string}'")
    if not _is_ast_safe(ast_tree):
        logger.error(f"AST validation failed: Expression '{expression_string}' includes unsupported operations or unsafe code.")
        raise ValueError("Expression includes unsupported operations or unsafe code.")
    
    logger.info(f"AST for expression '{expression_string}' is valid and safe.")
    return ast_tree


def _compile_and_evaluate_ast(ast_tree: ast.AST):
    """
    Compiles a validated AST into a code object and then evaluates it in a
    restricted environment to get the numerical result.

    Args:
        ast_tree: A validated (e.g., by `_parse_and_validate_ast`) ast.AST object.

    Returns:
        The numerical result (int or float) of the evaluated expression.

    Raises:
        ValueError: If evaluation fails due to issues like division by zero,
                    numerical overflow, the result not being a number, or other
                    runtime errors during evaluation.
    """
    try:
        # Compile the AST into a code object.
        # The filename '<string>' is a convention for code compiled from a string.
        code_object = compile(ast_tree, filename='<string>', mode='eval')
        
        # Evaluate the compiled code object.
        # A restricted global scope is provided:
        #   {"__builtins__": {}}  - This effectively removes access to all built-in
        #                           functions and variables, enhancing safety.
        #                           Only the operations defined in the AST (like Add, Sub)
        #                           and literals will be processed.
        logger.info("Evaluating compiled AST.")
        result = eval(code_object, {"__builtins__": {}})
        
        # After evaluation, ensure the result is a number (integer or float).
        # This handles cases where an expression might evaluate to something non-numeric
        # (e.g., if "None" or "True" were allowed and evaluated, though _is_ast_safe
        # and the final result check aim to prevent this outcome for typical math).
        if not isinstance(result, (int, float)):
            logger.error(f"Evaluation resulted in a non-numeric value: {result} (type: {type(result)})")
            raise ValueError("Evaluation resulted in a non-numeric value.")
        
        logger.info(f"AST evaluation successful. Result: {result}")
        return result
        
    except ZeroDivisionError as e:
        # Specifically catch division by zero to provide a clear error message.
        logger.error(f"Evaluation error: Division by zero. Details: {e}", exc_info=True)
        raise ValueError("Division by zero.")
    except OverflowError as e:
        # Catch numerical results that are too large to be represented.
        logger.error(f"Evaluation error: Numerical result out of range. Details: {e}", exc_info=True)
        raise ValueError("Numerical result out of range.")
    except Exception as e:
        # Catch any other unexpected errors during the compilation or evaluation phase.
        # This provides a generic fallback for other evaluation-time issues.
        logger.error(f"Unexpected error during AST compilation or evaluation. Details: {e}", exc_info=True)
        raise ValueError(f"Failed to evaluate expression: {e}")


def evaluate_expression(math_expr: str):
    """
    Evaluates a mathematical expression string and returns the result.

    The function supports basic arithmetic operations: addition (+), subtraction (-),
    multiplication (*), and division (/). It also handles parentheses for grouping.
    It employs a multi-stage process:
    1. Input Validation: Checks type, emptiness, and allowed characters.
    2. Secure Parsing: Converts the string to an Abstract Syntax Tree (AST) and
       validates the AST to allow only safe, basic arithmetic operations.
    3. Safe Evaluation: Compiles and evaluates the AST in a restricted environment.

    Args:
        math_expr: str, a string containing a mathematical expression to evaluate.

    Returns:
        The evaluated result of the expression (int or float).

    Raises:
        TypeError: If the input expression is not a string.
        ValueError: If the expression is empty, contains unsupported characters,
                    includes unsupported operations (e.g., exponentiation, bitwise ops),
                    constitutes unsafe code (e.g., function calls, attribute access),
                    has invalid syntax, or if evaluation leads to an error
                    (e.g., division by zero, overflow).
    """
    logger.info(f"Attempting to evaluate expression: '{math_expr}'")
    try:
        # Stage 1: Validate the input string.
        # This checks for type, emptiness, and allowed characters.
        logger.info("Starting input validation stage.")
        validated_expr_string = _validate_input_expression(math_expr)
        logger.info("Input validation successful.")
        
        # Stage 2: Parse the string into an AST and validate the AST for safety.
        # This ensures no malicious or unsupported operations are present.
        logger.info("Starting AST parsing and validation stage.")
        safe_ast_tree = _parse_and_validate_ast(validated_expr_string)
        logger.info("AST parsing and validation successful.")
        
        # Stage 3: Compile the safe AST and evaluate it.
        # This computes the final numerical result.
        logger.info("Starting AST compilation and evaluation stage.")
        result = _compile_and_evaluate_ast(safe_ast_tree)
        logger.info(f"Expression '{math_expr}' evaluated successfully. Result: {result}")
        
        return result
    except (TypeError, ValueError) as e:
        # Log the exception at this top level before re-raising.
        # The helper functions already log specifics, this gives a top-level view of failure.
        logger.error(f"Evaluation failed for expression '{math_expr}'. Error: {e}", exc_info=True)
        raise # Re-raise the caught exception (TypeError or ValueError)
    except Exception as e:
        # Catch any other unexpected exceptions not already handled.
        logger.critical(f"Unexpected critical error during evaluation of '{math_expr}'. Error: {e}", exc_info=True)
        # It's good practice to raise a consistent error type from the public API if possible,
        # or re-raise the original if its type is meaningful to the caller.
        # Here, we'll wrap it in a ValueError to maintain consistency with other error conditions.
        raise ValueError(f"An unexpected critical error occurred: {e}")
