import re

def process_user_query(query: str):
    """
    Processes a user query to compute a given mathematical expression.

    Args:
        query: str, the user-provided mathematical expression.

    Returns:
        The computed result of the expression.

    Raises:
        ValueError: if the input contains unsafe characters or if the expression is malformed.
    """
    # Validate the query to allow only numbers, operators, parentheses, and whitespace
    # This is a basic sanitization. For a production system, a more robust parser/evaluator is recommended.
    if not re.match(r"^[0-9\s\.\+\-\*\/\(\)]+$", query):
        raise ValueError("Input contains unsafe characters.")

    # Remove all whitespace to simplify further checks and evaluation
    sanitized_query = "".join(query.split())

    if not sanitized_query:
        raise ValueError("Expression cannot be empty.")

    try:
        # Using eval() can be risky if the input is not properly sanitized.
        # The regex above provides a basic level of safety.
        result = eval(sanitized_query)
        return result
    except SyntaxError:
        raise ValueError("Malformed expression: Syntax error.")
    except ZeroDivisionError:
        raise ValueError("Malformed expression: Division by zero.")
    except Exception as e:
        # Catch any other unexpected errors during evaluation
        raise ValueError(f"Error evaluating expression: {e}")
