

import re

def is_valid_expression(expression_string: str) -> bool:
    """
    Checks if a given string contains only numbers (including decimals),
    basic arithmetic operators (+, -, *, /), parentheses, and spaces.
    Prevents any form of code execution or access to external modules or attributes
    by strictly limiting the allowed characters.

    Args:
        expression_string: The string to validate.

    Returns:
        True if the input string consists only of allowed characters, False otherwise.
    """
    # An empty string is not considered a valid expression for calculation.
    if not expression_string:
        return False

    # Regex pattern:
    # ^ : Asserts position at the start of the string.
    # [ ... ]+ : Matches one or more occurrences of any character within the brackets.
    #   0-9 : Allows any digit.
    #   \.  : Allows a literal dot (for decimal points).
    #   \s  : Allows any whitespace character (e.g., space, tab).
    #   \+  : Allows a literal plus sign.
    #   \-  : Allows a literal minus sign.
    #   \*  : Allows a literal asterisk (for multiplication).
    #   \/  : Allows a literal forward slash (for division).
    #   \(  : Allows a literal opening parenthesis.
    #   \)  : Allows a literal closing parenthesis.
    # $ : Asserts position at the end of the string.
    #
    # This regex ensures that the entire string, from start to finish,
    # consists *only* of the characters specified. It does not validate
    # mathematical correctness (e.g., "5 * / 2" would pass this character check
    # but is not a valid mathematical expression).
    allowed_chars_pattern = r"^[0-9\.\s\+\-\*/\(\)]+$"

    if re.fullmatch(allowed_chars_pattern, expression_string):
        return True
    else:
        return False

def apply_operator(num1: float, num2: float, operator: str) -> float:
    """
    Applies the given arithmetic operator to two numbers.

    Args:
        num1: The first number (operand).
        num2: The second number (operand).
        operator: The arithmetic operator string ('+', '-', '*', '/').

    Returns:
        The result of applying the operator to num1 and num2.

    Raises:
        ValueError: If the operator is not one of '+', '-', '*', '/'.
        ZeroDivisionError: If attempting to divide by zero.
    """
    if operator == '+':
        return num1 + num2
    elif operator == '-':
        return num1 - num2
    elif operator == '*':
        return num1 * num2
    elif operator == '/':
        if num2 == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return num1 / num2
    else:
        raise ValueError(f"Unsupported operator: {operator}")

def compute_with_precedence(tokens: list) -> float:
    """
    Computes the result of an expression provided as a list of numbers and operators,
    respecting operator precedence (* and / before + and -).
    Assumes tokens is a list like [number, operator, number, ..., number].

    Args:
        tokens: A list of alternating numbers (float or int) and operator strings.
                Example: [2.0, '+', 3.0, '*', 4.0]

    Returns:
        The final computed result as a float.

    Raises:
        ValueError: If the token list is empty, malformed, or contains unsupported operators.
        ZeroDivisionError: If division by zero occurs.
    """
    if not tokens:
        raise ValueError("Input token list cannot be empty.")

    if len(tokens) == 1:
        if isinstance(tokens[0], (int, float)):
            return float(tokens[0])
        else:
            raise ValueError("Single token expression must be a number.")

    # Validate token structure: should be num, op, num, op, ... num
    # For a list of length N, there should be (N+1)/2 numbers and (N-1)/2 operators. N must be odd.
    if len(tokens) % 2 == 0:
        raise ValueError(f"Token list must have an odd number of elements (num, op, num...). Received {len(tokens)} elements.")
    for i, token in enumerate(tokens):
        is_even_idx = (i % 2 == 0) # 0-indexed: 0, 2, 4... are numbers
        if is_even_idx: # Number expected
            if not isinstance(token, (int, float)):
                raise ValueError(f"Expected number at token index {i}, got '{token}' (type: {type(token).__name__}).")
        else: # Operator expected
            if not isinstance(token, str):
                raise ValueError(f"Expected operator string at token index {i}, got '{token}' (type: {type(token).__name__}).")
            if token not in ('+', '-', '*', '/'):
                # This check can also be left to apply_operator or final list check,
                # but catching early for known unsupported ops is good.
                # However, the current loops only process valid ops, so unknown ops will lead to
                # the final "Expression could not be reduced" error. This is acceptable.
                pass


    ops_list = [float(t) if isinstance(t, (int, float)) else t for t in tokens]

    # Pass 1: Multiplication and Division
    i = 0
    while i < len(ops_list):
        token = ops_list[i]
        if isinstance(token, str) and token in ('*', '/'):
            # Operands are ops_list[i-1] and ops_list[i+1]
            # Validation of their existence and type is implicitly covered by initial checks
            # and the fact that operators must be at odd indices.
            num1 = ops_list[i-1] # Already float from list comprehension or previous op
            num2 = ops_list[i+1] # Already float
            
            result = apply_operator(num1, num2, token)
            ops_list = ops_list[:i-1] + [result] + ops_list[i+2:]
            i = i - 1  # Adjust index due to list modification
        else:
            i += 1
            
    # Pass 2: Addition and Subtraction
    i = 0
    while i < len(ops_list):
        token = ops_list[i]
        if isinstance(token, str) and token in ('+', '-'):
            num1 = ops_list[i-1]
            num2 = ops_list[i+1]

            result = apply_operator(num1, num2, token)
            ops_list = ops_list[:i-1] + [result] + ops_list[i+2:]
            i = i - 1 
        else:
            i += 1
            
    if len(ops_list) == 1 and isinstance(ops_list[0], float):
        return ops_list[0]
    else:
        # This can happen if the list was not a valid sequence of ops and nums
        # or if it contained unknown elements that weren't processed.
        raise ValueError(f"Expression could not be reduced to a single number. Final list: {ops_list}")


if __name__ == '__main__':
    # Test cases for is_valid_expression
    print("--- Testing is_valid_expression ---")
    # Valid cases (handles decimals)
    print(f"'1 + 1': {is_valid_expression('1 + 1')}")
    print(f"'2 * (3 - 1) / 4': {is_valid_expression('2 * (3 - 1) / 4')}")
    print(f"'100-23*4': {is_valid_expression('100-23*4')}")
    print(f"'(5+5)*2': {is_valid_expression('(5+5)*2')}")
    print(f"'   123   ': {is_valid_expression('   123   ')}")
    print(f"'1.0 + 2.5': {is_valid_expression('1.0 + 2.5')}")
    print(f"'.5 * 2.': {is_valid_expression('.5 * 2.')}")
    print(f"'0.5 + .5': {is_valid_expression('0.5 + .5')}")


    # Invalid cases for is_valid_expression
    print(f"Empty string '': {is_valid_expression('')}")
    print(f"'1 + 1; drop table users': {is_valid_expression('1 + 1; drop table users')}")
    s_injection1 = 'import os; os.system("clear")'
    print(f"'{s_injection1}': {is_valid_expression(s_injection1)}")
    s_injection2 = 'eval("1+1")'
    print(f"'{s_injection2}': {is_valid_expression(s_injection2)}")
    s_injection3 = '__import__("os").getcwd()'
    print(f"'{s_injection3}': {is_valid_expression(s_injection3)}")
    print(f"'5 % 2': {is_valid_expression('5 % 2')}")
    print(f"'5 ^ 2': {is_valid_expression('5 ^ 2')}")
    print(f"'alpha + 1': {is_valid_expression('alpha + 1')}")
    print(f"'1+': {is_valid_expression('1+')}") 
    print(f"'*2': {is_valid_expression('*2')}")


    # Test cases for apply_operator
    print("\n--- Testing apply_operator ---")
    print(f"3.0 + 5.0 = {apply_operator(3.0, 5.0, '+')}")
    print(f"10.0 - 4.0 = {apply_operator(10.0, 4.0, '-')}")
    print(f"6.0 * 7.0 = {apply_operator(6.0, 7.0, '*')}")
    print(f"8.0 / 2.0 = {apply_operator(8.0, 2.0, '/')}")
    print(f"5.0 / 2.0 = {apply_operator(5.0, 2.0, '/')}")
    try:
        apply_operator(1.0, 0.0, '/')
    except ZeroDivisionError as e:
        print(f"1.0 / 0.0: Caught expected error: {e}")
    try:
        apply_operator(1.0, 1.0, '%')
    except ValueError as e:
        print(f"1.0 % 1.0: Caught expected error: {e}")

    # Test cases for compute_with_precedence
    print("\n--- Testing compute_with_precedence ---")
    print(f"compute_with_precedence([1.0, '+', 1.0]): {compute_with_precedence([1.0, '+', 1.0])}")
    print(f"compute_with_precedence([2.0, '*', 3.0, '+', 4.0]): {compute_with_precedence([2.0, '*', 3.0, '+', 4.0])}")
    print(f"compute_with_precedence([2.0, '+', 3.0, '*', 4.0]): {compute_with_precedence([2.0, '+', 3.0, '*', 4.0])}")
    print(f"compute_with_precedence([10.0, '-', 2.0, '*', 3.0]): {compute_with_precedence([10.0, '-', 2.0, '*', 3.0])}")
    print(f"compute_with_precedence([10.0, '*', 2.0, '-', 3.0]): {compute_with_precedence([10.0, '*', 2.0, '-', 3.0])}")
    print(f"compute_with_precedence([10.0, '/', 2.0, '+', 3.0, '*', 4.0, '-', 5.0]): {compute_with_precedence([10.0, '/', 2.0, '+', 3.0, '*', 4.0, '-', 5.0])}")
    print(f"compute_with_precedence([5.0]): {compute_with_precedence([5.0])}")
    print(f"compute_with_precedence([3.5]): {compute_with_precedence([3.5])}")
    print(f"compute_with_precedence([2.0, '*', 3.0, '*', 4.0]): {compute_with_precedence([2.0, '*', 3.0, '*', 4.0])}")
    print(f"compute_with_precedence([10.0, '/', 2.0, '/', 5.0]): {compute_with_precedence([10.0, '/', 2.0, '/', 5.0])}") # 10/2=5, 5/5=1

    # Error cases for compute_with_precedence
    print("\nError cases for compute_with_precedence:")
    test_cases_errors = [
        ([], "Input token list cannot be empty."),
        (['a'], "Single token expression must be a number."),
        ([1.0, '+'], "Token list must have an odd number of elements (num, op, num...). Received 2 elements."),
        ([1.0, '+', 'a'], "Expected number at token index 2, got 'a' (type: str)."),
        ([1.0, '%', 2.0], "Expression could not be reduced to a single number. Final list: [1.0, '%', 2.0]"), # '%' is not processed
        (['+', 1.0, 2.0], "Expected number at token index 0, got '+' (type: str)."),
        ([1.0, 2.0, '+'], "Expected operator string at token index 1, got '2.0' (type: float)."),
        ([1.0, '+', 2.0, 3.0], "Token list must have an odd number of elements (num, op, num...). Received 4 elements.")
    ]

    for tokens_input, expected_msg_part in test_cases_errors:
        try:
            compute_with_precedence(list(tokens_input)) # Pass a copy
            print(f"compute_with_precedence({tokens_input}): Expected ValueError, but none was raised.")
        except ValueError as e:
            if expected_msg_part in str(e):
                print(f"compute_with_precedence({tokens_input}): Caught expected error: {e}")
            else:
                print(f"compute_with_precedence({tokens_input}): Caught ValueError, but message mismatch. Got: '{e}', Expected part: '{expected_msg_part}'")
        except Exception as e:
            print(f"compute_with_precedence({tokens_input}): Caught unexpected error type: {type(e).__name__} - {e}")


    try:
        compute_with_precedence([1.0, '/', 0.0])
    except ZeroDivisionError as e:
        print(f"compute_with_precedence([1.0, '/', 0.0]): Caught expected error: {e}")
    except Exception as e:
        print(f"compute_with_precedence([1.0, '/', 0.0]): Caught unexpected error type for division by zero: {type(e).__name__} - {e}")
