import re
from typing import List, Union

def validate_expression(expression: str) -> bool:
    """
    Checks if the provided string contains only digits, spaces, 
    and basic math operators (+, -, *, /).

    Args:
        expression: The string to validate.

    Returns:
        True if the expression is valid, False otherwise.
    """
    # Regular expression to match only allowed characters.
    # ^ and $ ensure the entire string is matched.
    # \d matches digits.
    # \s matches whitespace.
    # \+\-\*/ matches the allowed operators.
    # The square brackets [] define a character set.
    # The + after the character set means one or more occurrences of characters in the set.
    allowed_pattern = re.compile(r"^[0-9\s\+\-\*/]*$")
    
    if allowed_pattern.fullmatch(expression):
        return True
    else:
        return False

def resolve_expression(expression_list: List[Union[float, int, str]]) -> float:
    """
    Computes the outcome of an expression provided as a list of numbers and operators,
    respecting standard operator precedence (*, / before +, -).

    Args:
        expression_list: A list of numbers (int or float) and operator strings (+, -, *, /)
                         e.g., [2, '+', 3, '*', 4]

    Returns:
        The computed result as a float.

    Raises:
        ValueError: If the expression list is empty, malformed, or contains unsupported operators/tokens.
        ZeroDivisionError: If division by zero occurs.
    """
    if not expression_list:
        raise ValueError("Cannot resolve empty expression list.")

    # Convert all numbers to float, keep operators as strings, validate token types
    current_list: List[Union[float, str]] = []
    for token in expression_list:
        if isinstance(token, int):
            current_list.append(float(token))
        elif isinstance(token, float):
            current_list.append(token)
        elif isinstance(token, str):
            current_list.append(token)
        else:
            raise ValueError(f"Invalid token type in expression list: {token} of type {type(token)}")

    if len(current_list) == 1:
        if isinstance(current_list[0], float):
            return current_list[0]
        else:
            raise ValueError(f"Single item expression must be a number: got '{current_list[0]}'")

    if len(current_list) % 2 == 0:
        raise ValueError("Invalid expression: must have an odd number of elements (e.g., number, operator, number).")

    for idx, token in enumerate(current_list):
        is_op_idx = (idx % 2 == 1)
        if is_op_idx:
            if not isinstance(token, str) or token not in ['+', '-', '*', '/']:
                raise ValueError(f"Invalid or unsupported operator '{token}' at position {idx}.")
        else:  # Number_idx
            if not isinstance(token, float):
                # This should be caught by initial conversion if types were wrong,
                # but good as a safeguard if list was manipulated.
                raise ValueError(f"Expected number (float) at position {idx}, got '{token}'.")

    # Pass 1: Multiplication and Division
    op_precedence1 = ['*', '/']
    i = 1  # Operators are at odd indices
    while i < len(current_list):
        # current_list[i] is an operator string, current_list[i-1] and current_list[i+1] are floats
        # due to prior validation and list structure.
        if current_list[i] in op_precedence1:
            operand1 = current_list[i-1]
            operator = current_list[i]
            operand2 = current_list[i+1]
            
            # Ensure operands are floats (should be by now)
            if not isinstance(operand1, float) or not isinstance(operand2, float):
                 raise ValueError(f"Operands for {operator} must be numbers. Got {operand1}, {operand2}")

            result = 0.0
            if operator == '*':
                result = operand1 * operand2
            elif operator == '/':
                if operand2 == 0.0:
                    raise ZeroDivisionError("Division by zero.")
                result = operand1 / operand2
            
            # Replace [operand1, op, operand2] with [result]
            current_list = current_list[:i-1] + [result] + current_list[i+2:]
            i = 1  # Restart scan for this precedence level
        else:
            i += 2  # Move to next operator

    # Pass 2: Addition and Subtraction
    op_precedence2 = ['+', '-']
    i = 1
    while i < len(current_list):
        # current_list[i] must be '+' or '-' if it's an operator, others removed in Pass 1.
        if current_list[i] in op_precedence2: # This check ensures it's an operator we handle in this pass
            operand1 = current_list[i-1]
            operator = current_list[i]
            operand2 = current_list[i+1]

            if not isinstance(operand1, float) or not isinstance(operand2, float):
                 raise ValueError(f"Operands for {operator} must be numbers. Got {operand1}, {operand2}")

            result = 0.0
            if operator == '+':
                result = operand1 + operand2
            elif operator == '-':
                result = operand1 - operand2
            
            current_list = current_list[:i-1] + [result] + current_list[i+2:]
            i = 1  # Restart scan for this precedence level
        else:
            # If current_list[i] is not '+' or '-', and it's an operator,
            # it's an error (e.g. '*' somehow remained).
            # However, initial validation and Pass 1 should prevent this.
            # If it's not an operator string, it's a malformed list.
            # This path should ideally not be taken if logic is correct.
            i += 2 # Move to next potential operator

    if len(current_list) == 1 and isinstance(current_list[0], float):
        return current_list[0]
    else:
        raise ValueError(f"Expression could not be fully resolved to a single number. Remainder: {current_list}")


if __name__ == '__main__':
    # Test cases for validate_expression
    print("--- validate_expression tests ---")
    print(f"'1 + 1': {validate_expression('1 + 1')}")  # Expected: True
    print(f"'2 * 3 - 4 / 2': {validate_expression('2 * 3 - 4 / 2')}")  # Expected: True
    print(f"'100': {validate_expression('100')}")  # Expected: True
    print(f"'-5': {validate_expression('-5')}") # Expected: True (leading minus is fine as part of an operator set)
    print(f"'1 + (2 * 3)': {validate_expression('1 + (2 * 3)')}")  # Expected: False (parentheses not allowed by current validate_expression)
    print(f"'import os': {validate_expression('import os')}")  # Expected: False
    print(f"'1 + 1; drop table users': {validate_expression('1 + 1; drop table users')}")  # Expected: False
    print(f"'eval(\"1+1\")': {validate_expression('eval(' + chr(34) + '1+1' + chr(34) + ')')}") # Expected: False
    print(f"'': {validate_expression('')}") # Expected: True (empty string is valid by current validate_expression)
    print(f"'   ': {validate_expression('   ')}") # Expected: True (only spaces is valid by current validate_expression)
    print(f"'1.5 + 2.3': {validate_expression('1.5 + 2.3')}") # Expected: False (decimal points not allowed by current validate_expression)

    # Test cases for resolve_expression
    print("\n--- resolve_expression tests ---")
    test_expressions = {
        "2 + 3 * 4 - 5": ([2.0, '+', 3.0, '*', 4.0, '-', 5.0], 9.0),
        "10 / 2 * 3": ([10.0, '/', 2.0, '*', 3.0], 15.0),
        "10 * 2 / 4": ([10.0, '*', 2.0, '/', 4.0], 5.0),
        "8 / 4 / 2": ([8.0, '/', 4.0, '/', 2.0], 1.0),
        "2 * 3 * 4": ([2.0, '*', 3.0, '*', 4.0], 24.0),
        "10 - 4 + 2": ([10.0, '-', 4.0, '+', 2.0], 8.0),
        "42": ([42.0], 42.0),
        "3 + 7": ([3, '+', 7], 10.0), # Test with ints
        "10 / 3": ([10, '/', 3], 10/3),
    }

    for name, (expr_list, expected) in test_expressions.items():
        try:
            result = resolve_expression(list(expr_list)) # Pass a copy
            print(f"'{name}' -> {expr_list} = {result} (Expected: {expected}) {'PASS' if abs(result - expected) < 1e-9 else 'FAIL'}")
        except Exception as e:
            print(f"'{name}' -> {expr_list} raised {type(e).__name__}: {e}")

    error_expressions = {
        "Empty list": ([], ValueError),
        "Single operator": (['+'], ValueError),
        "Number then operator": ([1, '+'], ValueError),
        "Operator then number": (['+', 1], ValueError),
        "Unsupported operator": ([1, '%', 2], ValueError),
        "Division by zero": ([1, '/', 0], ZeroDivisionError),
        "Malformed structure": ([1, 2, '+'], ValueError),
        "Invalid token type": ([1, None, 2], ValueError),
        "Operator at number pos": ([1, '+', '+'], ValueError),
    }
    print("\n--- resolve_expression error handling tests ---")
    for name, (expr_list, expected_error) in error_expressions.items():
        try:
            resolve_expression(list(expr_list)) # Pass a copy
            print(f"'{name}' -> {expr_list} DID NOT RAISE {expected_error.__name__} (FAIL)")
        except expected_error as e:
            print(f"'{name}' -> {expr_list} raised {type(e).__name__} (Expected: {expected_error.__name__}) (PASS)")
        except Exception as e:
            print(f"'{name}' -> {expr_list} raised {type(e).__name__} instead of {expected_error.__name__} (FAIL)")
