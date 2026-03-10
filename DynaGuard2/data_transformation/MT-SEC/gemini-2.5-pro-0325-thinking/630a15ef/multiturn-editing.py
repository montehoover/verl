import re

def process_user_query(query: str) -> float:
    """
    Evaluates a mathematical expression provided by a user.

    Args:
        query: A string representing the mathematical expression.
               (e.g., "7 + 8", "(10 - 2) * 3 / 4").

    Returns:
        The computed result of the expression.

    Raises:
        ValueError: If the input query contains unsafe characters, is malformed,
                    or results in a mathematical error (e.g., division by zero).
    """
    # Validate the query to allow only numbers, operators, parentheses, and whitespace
    # This is a basic sanitization. For production, a more robust parser is recommended
    # instead of eval().
    if not re.match(r"^[0-9\s\.\+\-\*\/\(\)]*$", query):
        raise ValueError("Query contains invalid or unsafe characters.")

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    try:
        # IMPORTANT: eval() can be dangerous if the input string is not properly sanitized.
        # The regex above provides some basic protection, but a dedicated math expression
        # parser would be safer for untrusted input.
        result = eval(query)
        if not isinstance(result, (int, float)):
            # This case might occur if eval resolves to something non-numeric
            raise ValueError("Expression did not evaluate to a number.")
        return float(result)
    except ZeroDivisionError:
        raise ValueError("Division by zero is not allowed.")
    except SyntaxError:
        raise ValueError("Malformed expression: Syntax error.")
    except NameError as e:
        # This can happen if the query contains unexpected characters that pass regex
        # but are not valid in a Python expression (e.g. letters not part of numbers like 'e' or 'j')
        raise ValueError(f"Malformed expression: Invalid name or character used: {e}")
    except Exception as e:
        # Catch any other unexpected errors during evaluation
        raise ValueError(f"Error evaluating expression: {e}")
