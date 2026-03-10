import re

def validate_expression(expression_string: str) -> bool:
    """
    Validates if the input string consists solely of digits, basic
    arithmetic operators (+, -, *, /), parentheses, and spaces.

    Args:
        expression_string: The string to validate.

    Returns:
        True if the string is valid, False otherwise.
    """
    # Pattern explanation:
    # ^ : Asserts position at the start of the string.
    # [ ... ] : Character set. Matches any single character within the brackets.
    #   0-9 : Matches any digit from 0 to 9.
    #   \s : Matches any whitespace character (this was changed to a literal space ' ' based on refinement).
    #        Using ' ' (literal space) to only allow space character, not tabs/newlines etc.
    #   \+ : Matches the plus sign.
    #   \- : Matches the minus sign.
    #   \* : Matches the asterisk (multiplication sign).
    #   \/ : Matches the forward slash (division sign).
    #   \( : Matches the opening parenthesis.
    #   \) : Matches the closing parenthesis.
    #   Inside a character set [], most special characters (like +, *, (, )) lose their special meaning
    #   and can be used literally. However, - needs to be handled carefully (e.g., at the end or escaped)
    #   if not defining a range.
    #   The pattern [0-9 +\-*/()] is a more concise way to write this for the character set.
    # * : Matches the preceding element (the character set) zero or more times.
    # $ : Asserts position at the end of the string.
    #
    # So, the entire pattern ensures that the string, from start to end,
    # consists only of the allowed characters.
    pattern = r"^[0-9 +\-*/()]*$"
    if re.fullmatch(pattern, expression_string):
        return True
    return False


def apply_operator(num1: float, num2: float, operator_str: str) -> float:
    """
    Applies a single arithmetic operator to two numbers.

    Args:
        num1: The first number (operand).
        num2: The second number (operand).
        operator_str: The operator ('+', '-', '*', '/') as a string.

    Returns:
        The result of applying the operator.

    Raises:
        ValueError: If the operator_str is not one of '+', '-', '*', '/'.
        ZeroDivisionError: If attempting to divide by zero.
    """
    if operator_str == '+':
        return num1 + num2
    elif operator_str == '-':
        return num1 - num2
    elif operator_str == '*':
        return num1 * num2
    elif operator_str == '/':
        if num2 == 0:
            raise ZeroDivisionError("Division by zero.")
        return num1 / num2
    else:
        raise ValueError(f"Unsupported operator: {operator_str}")


def calculate_with_precedence(tokens: list) -> float:
    """
    Calculates the result of an arithmetic expression represented by a list
    of numbers (float) and operators (str), respecting operator precedence
    (* and / before + and -).

    Args:
        tokens: A list of alternating numbers and operators.
                Example: [2.0, '+', 3.0, '*', 4.0]

    Returns:
        The final calculated result.

    Raises:
        ValueError: If an unsupported operator is encountered or the token list
                    is malformed (e.g., not enough operands for an operator).
        TypeError: If tokens are not of expected types (float for numbers, str for ops).
        ZeroDivisionError: If division by zero occurs.
    """
    if not tokens:
        raise ValueError("Input token list cannot be empty.")

    # Make a copy to avoid modifying the original list if passed by reference
    # and to ensure numbers are floats.
    processed_tokens = []
    for token in tokens:
        if isinstance(token, (int, float)):
            processed_tokens.append(float(token))
        elif isinstance(token, str):
            processed_tokens.append(token)
        else:
            raise TypeError(f"Invalid token type: {type(token)}. Expected number or operator string.")

    # Pass 1: Handle * and /
    i = 0
    while i < len(processed_tokens):
        token = processed_tokens[i]
        if token == '*' or token == '/':
            if i == 0 or i == len(processed_tokens) - 1:
                raise ValueError("Operator at invalid position.")
            num1 = processed_tokens[i-1]
            num2 = processed_tokens[i+1]
            if not isinstance(num1, float) or not isinstance(num2, float):
                raise ValueError("Invalid sequence: operator must be between two numbers.")

            result = apply_operator(num1, num2, token)
            # Replace num1, operator, num2 with the result
            processed_tokens = processed_tokens[:i-1] + [result] + processed_tokens[i+2:]
            i = i - 1 # Adjust index to re-evaluate from the new result's position
        else:
            i += 1

    # Pass 2: Handle + and -
    i = 0
    while i < len(processed_tokens):
        token = processed_tokens[i]
        if token == '+' or token == '-':
            if i == 0 or i == len(processed_tokens) - 1:
                # This could be a leading minus/plus, but for a simple list of [num, op, num, ...]
                # it's an invalid position for a binary operator.
                # For simplicity, this calculator assumes binary operations.
                # A more complex parser would handle unary minus.
                raise ValueError("Operator at invalid position for binary operation.")
            num1 = processed_tokens[i-1]
            num2 = processed_tokens[i+1]
            if not isinstance(num1, float) or not isinstance(num2, float):
                raise ValueError("Invalid sequence: operator must be between two numbers.")

            result = apply_operator(num1, num2, token)
            processed_tokens = processed_tokens[:i-1] + [result] + processed_tokens[i+2:]
            i = i - 1
        else:
            i += 1
    
    if len(processed_tokens) == 1 and isinstance(processed_tokens[0], float):
        return processed_tokens[0]
    elif not processed_tokens: # Should not happen if input was not empty
        raise ValueError("Calculation resulted in an empty token list.")
    else:
        # This can happen if the input was just a number, or if there's a logic error
        # or malformed input not caught earlier.
        # If input was just [5.0], it should pass through both loops and be returned.
        # If input was ['+'], it would be caught by operator position checks.
        raise ValueError(f"Calculation did not resolve to a single number. Remaining tokens: {processed_tokens}")
