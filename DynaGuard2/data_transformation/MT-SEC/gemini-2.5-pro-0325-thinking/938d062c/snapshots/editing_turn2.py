import re

def simple_calculator(expression: str) -> float:
    """
    Evaluates a mathematical expression string.

    Args:
        expression: The mathematical expression string (e.g., "4 + 5 * (2 - 1)").
                    Supports addition, subtraction, multiplication, division,
                    and parentheses for precedence. Numbers can be integers or floats.

    Returns:
        The result of the expression as a float.

    Raises:
        ValueError: If the expression string contains invalid characters,
                    is malformed, or does not evaluate to a number.
        ZeroDivisionError: If division by zero is attempted within the expression.
    """
    # Sanitize the expression to allow only numbers, operators, parentheses, and spaces.
    # Numbers can be integers or floats (e.g., "3", "3.14").
    # Operators: +, -, *, /
    # Parentheses: ( )
    # Whitespace is allowed and will be handled by eval().
    allowed_chars_pattern = r"^[0-9\.\+\-\*\/\(\)\s]+$"
    if not re.match(allowed_chars_pattern, expression.strip()):
        # Check expression.strip() to ensure an empty or all-whitespace string is caught
        # if it wasn't already by the regex due to an issue or if it's empty.
        # An empty string or one with only whitespace will not match if `+` is used in regex.
        # If expression.strip() is empty, it means original was empty or all whitespace.
        if not expression.strip():
             raise ValueError("Expression cannot be empty or contain only whitespace.")
        raise ValueError(
            "Expression contains invalid characters. "
            "Only numbers, operators (+, -, *, /), parentheses, and spaces are allowed."
        )

    try:
        # Using eval() for its ability to parse and calculate expressions
        # respecting operator precedence and parentheses.
        # The sanitization step above provides a basic layer of security by restricting
        # the character set, mitigating common eval-related risks.
        # For environments requiring higher security with untrusted input,
        # consider using a dedicated and safer expression evaluation library
        # (e.g., asteval) or implementing a custom parser.
        
        # Ensure the expression is not empty after stripping, to avoid eval issues.
        if not expression.strip():
            raise ValueError("Expression is empty or all whitespace.")

        result = eval(expression)

        if not isinstance(result, (int, float)):
            # This check is a safeguard in case eval produces an unexpected type,
            # though the regex and nature of arithmetic operations should lead to numbers.
            raise ValueError("Expression did not evaluate to a numerical result.")
        
        return float(result)
    except SyntaxError:
        raise ValueError("Invalid expression syntax. Check for malformed parts or unbalanced parentheses.")
    except ZeroDivisionError:
        # This is raised by Python during eval if division by zero occurs.
        raise ZeroDivisionError("Cannot divide by zero in the expression.")
    except Exception as e:
        # Catch other potential errors during eval, e.g., NameError if somehow
        # a variable name slipped through (though regex should prevent this).
        # This provides a generic fallback.
        raise ValueError(f"Error evaluating expression: {str(e)}")
