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
    #   .   : Matches a literal dot (for decimal points).
    #   eE  : Matches 'e' or 'E' (for scientific notation).
    #       : Matches a literal space.
    #   +\-*/() : Matches basic arithmetic operators and parentheses.
    # * : Matches the preceding element (the character set) zero or more times.
    # $ : Asserts position at the end of the string.
    #
    # So, the entire pattern ensures that the string, from start to end,
    # consists only of the allowed characters for a typical calculator expression.
    pattern = r"^[0-9 .eE+\-*/()]*$"
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


# Setup code provided by user
import ast

def parse_and_calculate(expression_string: str) -> float:
    """
    Parses an arithmetic expression string, validates it, handles parentheses,
    and calculates the result using 'calculate_with_precedence'.

    Args:
        expression_string: The arithmetic expression string.

    Returns:
        The calculated result as a float.

    Raises:
        ValueError: If the input string is invalid, contains unsupported characters,
                    is malformed, results in an undefined operation (like division by zero),
                    or leads to a non-numeric result (inf/nan).
    """
    if not validate_expression(expression_string):
        raise ValueError(f"Invalid characters in expression: '{expression_string}'")

    current_expr = expression_string.replace(" ", "")

    if not current_expr:
        raise ValueError("Expression is empty or contains only spaces.")

    # Handle parentheses by recursively evaluating sub-expressions
    while '(' in current_expr:
        end_index = current_expr.find(')')
        if end_index == -1:
            raise ValueError("Mismatched parentheses: missing ')'")

        # Find the matching opening parenthesis for the rightmost ')' found
        # This ensures we process innermost parentheses first for expressions like ((1+2)+3)
        start_index = current_expr.rfind('(', 0, end_index)
        if start_index == -1:
            raise ValueError("Mismatched parentheses: missing '('")

        sub_expression = current_expr[start_index + 1:end_index]

        if not sub_expression:
            raise ValueError("Empty parentheses '()' found in expression.")

        # Recursively calculate the value of the sub-expression
        sub_result = parse_and_calculate(sub_expression)

        # Replace the parenthesized sub-expression with its string result
        current_expr = (
            current_expr[:start_index] +
            str(sub_result) +
            current_expr[end_index + 1:]
        )

    # Tokenize the (now flat, no-spaces, no-parentheses) expression string
    # Regex supports:
    # - Optional leading minus for numbers (e.g., -5, -0.5, -.5)
    # - Decimal points (e.g., 3.14, .5)
    # - Scientific notation (e.g., 1e5, -2.5e-2, .5E10)
    # - Operators: +, -, *, /
    token_pattern = r"-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|[+\-*/]"
    raw_tokens = re.findall(token_pattern, current_expr)

    if not raw_tokens:
        if current_expr: # Non-empty string that couldn't be tokenized (e.g. ".", "1.2.3")
            raise ValueError(f"Invalid expression format: could not tokenize '{current_expr}'")
        else: # Should have been caught by earlier checks if current_expr became empty
            raise ValueError("Expression became empty and could not be tokenized.")

    final_tokens = []
    for i, token_str in enumerate(raw_tokens):
        if token_str in "+-*/":
            if i == 0:
                raise ValueError(f"Expression cannot start with operator '{token_str}'. Use numbers like '-3'.")
            if i == len(raw_tokens) - 1:
                raise ValueError(f"Expression cannot end with operator '{token_str}'.")
            if raw_tokens[i-1] in "+-*/": # Check for consecutive operators
                raise ValueError(f"Consecutive operators found: {raw_tokens[i-1]}{token_str}")
            final_tokens.append(token_str)
        else: # Token should be a number
            try:
                num_val = float(token_str)
                # Check for inf/nan early, as they can cause issues or are undesirable results
                if abs(num_val) == float('inf') or num_val != num_val: # num_val != num_val is true for NaN
                    raise ValueError("Number token results in infinity or NaN during parsing.")
                final_tokens.append(num_val)
            except ValueError:
                raise ValueError(f"Invalid number format in token: '{token_str}'")
    
    if not final_tokens : # Should be caught by raw_tokens check, but as a safeguard
        raise ValueError("Expression resulted in no valid tokens.")

    try:
        result = calculate_with_precedence(final_tokens)
        if abs(result) == float('inf') or result != result: # Check final result for inf/nan
            raise ValueError("Calculation resulted in infinity or NaN.")
        return result
    except ZeroDivisionError: # Catch division by zero from calculate_with_precedence
        raise ValueError("Division by zero.")
    # Other ValueErrors from calculate_with_precedence will propagate naturally.
