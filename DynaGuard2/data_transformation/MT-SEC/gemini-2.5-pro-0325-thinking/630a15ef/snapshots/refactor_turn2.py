import re

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
        raise ValueError(
            "Input contains unsafe characters. Only numbers, operators (+, -, *, /), "
            "parentheses, and decimal points are allowed."
        )

    # Remove all whitespace characters (spaces, tabs, newlines, etc.)
    # from the query to simplify parsing and evaluation.
    # Example: " 2 + 3 " becomes "2+3"
    sanitized_query = "".join(query.split())

    # Check if the query becomes empty after removing whitespace.
    # An empty expression is considered malformed.
    if not sanitized_query:
        raise ValueError("Expression cannot be empty or contain only whitespace.")

    try:
        # Evaluate the sanitized mathematical expression.
        # WARNING: eval() can execute arbitrary code if the input is not
        # strictly controlled. The regex validation above provides a basic
        # layer of safety, but it's not foolproof for all scenarios.
        # Consider safer alternatives for production code if complex expressions
        # or higher security is needed.
        result = eval(sanitized_query)
        return result
    except SyntaxError:
        # Raised if the expression has incorrect syntax.
        # Example: "2 + * 3" or "((3+2)"
        raise ValueError("Malformed expression: Syntax error.")
    except ZeroDivisionError:
        # Raised if the expression attempts to divide by zero.
        # Example: "5 / 0"
        raise ValueError("Malformed expression: Division by zero.")
    except Exception as e:
        # Catch any other unexpected errors during evaluation.
        # This could include TypeError for operations on incompatible types
        # if the regex somehow allowed such an expression to pass, though unlikely
        # with the current strict pattern.
        raise ValueError(f"Error evaluating expression: {e}")
