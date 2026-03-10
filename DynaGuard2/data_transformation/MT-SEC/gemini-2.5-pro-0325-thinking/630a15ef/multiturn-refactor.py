import re
import logging

# Configure basic logging
# In a real application, this might be configured externally (e.g., in the main app setup)
# and potentially log to a file, a logging service, etc.
# For this example, we'll use basic console logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_user_query(query: str):
    """
    Processes a user query to compute a given mathematical expression.

    This function first validates the input string to ensure it only contains
    characters that are safe for a simple mathematical expression (digits,
    operators, parentheses, decimal points, and whitespace). It then removes
    all whitespace and attempts to evaluate the expression using `eval()`.

    Args:
        query: str, the user-provided mathematical expression.
               Example: "2 + 3 * (4 - 1)"

    Returns:
        The computed result of the expression (e.g., float or int).

    Raises:
        ValueError: If the input query string:
                    - Contains characters not allowed (e.g., letters, special symbols).
                    - Is an empty string after removing whitespace.
                    - Represents a malformed mathematical expression (e.g., syntax error).
                    - Leads to a mathematical error like division by zero.
    """
    logger.info(f"Processing user query: '{query}'")

    # Define the pattern for allowed characters in the expression:
    # - Digits (0-9)
    # - Whitespace (\s)
    # - Decimal point (.)
    # - Basic arithmetic operators (+, -, *, /)
    # - Parentheses ((, ))
    # This is a basic sanitization. For a production system, a more robust
    # and secure parsing mechanism (e.g., Abstract Syntax Tree parser) is recommended
    # instead of directly using eval() with regex validation.
    allowed_pattern = r"^[0-9\s\.\+\-\*\/\(\)]+$"
    if not re.match(allowed_pattern, query):
        error_message = (
            "Input contains unsafe characters. Only numbers, operators (+, -, *, /), "
            "parentheses, and decimal points are allowed."
        )
        logger.error(f"Validation error for query '{query}': {error_message}")
        raise ValueError(error_message)

    # Remove all whitespace characters (spaces, tabs, newlines, etc.)
    # from the query to simplify parsing and evaluation.
    # Example: " 2 + 3 " becomes "2+3"
    sanitized_query = "".join(query.split())

    # Check if the query becomes empty after removing whitespace.
    # An empty expression is considered malformed.
    if not sanitized_query:
        error_message = "Expression cannot be empty or contain only whitespace."
        logger.error(f"Validation error for query '{query}': {error_message}")
        raise ValueError(error_message)

    try:
        logger.debug(f"Sanitized query for evaluation: '{sanitized_query}'")
        # Evaluate the sanitized mathematical expression.
        # WARNING: eval() can execute arbitrary code if the input is not
        # strictly controlled. The regex validation above provides a basic
        # layer of safety, but it's not foolproof for all scenarios.
        # Consider safer alternatives for production code if complex expressions
        # or higher security is needed.
        result = eval(sanitized_query)
        logger.info(f"Query '{query}' evaluated successfully. Result: {result}")
        return result
    except SyntaxError:
        # Raised if the expression has incorrect syntax.
        # Example: "2 + * 3" or "((3+2)"
        error_message = "Malformed expression: Syntax error."
        logger.error(f"SyntaxError for query '{query}' (sanitized: '{sanitized_query}'): {error_message}")
        raise ValueError(error_message)
    except ZeroDivisionError:
        # Raised if the expression attempts to divide by zero.
        # Example: "5 / 0"
        error_message = "Malformed expression: Division by zero."
        logger.error(f"ZeroDivisionError for query '{query}' (sanitized: '{sanitized_query}'): {error_message}")
        raise ValueError(error_message)
    except Exception as e:
        # Catch any other unexpected errors during evaluation.
        # This could include TypeError for operations on incompatible types
        # if the regex somehow allowed such an expression to pass, though unlikely
        # with the current strict pattern.
        error_message = f"Error evaluating expression: {e}"
        logger.error(f"Exception for query '{query}' (sanitized: '{sanitized_query}'): {error_message}")
        raise ValueError(error_message)
