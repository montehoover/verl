import re

def is_valid_expression(expression: str) -> bool:
    """
    Checks if a given string contains only numbers, basic arithmetic operators
    (+, -, *, /), parentheses, and spaces.

    Args:
        expression: The string to validate.

    Returns:
        True if the expression is valid, False otherwise.
    """
    # Regex to allow only numbers, operators (+, -, *, /), parentheses, and spaces.
    # ^ and $ ensure the entire string matches the pattern.
    # [\d\s()+\-*/]* matches zero or more occurrences of the allowed characters.
    # \d for digits
    # \s for whitespace
    # () for literal parentheses (escaped as \( and \))
    # +-*/ for literal operators (escaped as \+, \-, \*, \/)
    # Note: Inside a character set [], most characters don't need escaping,
    # but it's good practice for clarity or if they are at special positions (e.g., -).
    # For this specific set, only \ might need escaping if used literally.
    # - is special if not at the start or end, or not part of a range.
    # * and + are not special inside [].
    # / is not special.
    # ( and ) are not special inside [].
    # So, the pattern can be simplified.
    pattern = r"^[0-9\s()+\-*/]*$"
    if re.fullmatch(pattern, expression):
        return True
    return False

def apply_operator(operand1: float, operand2: float, operator: str) -> float:
    """
    Applies a single arithmetic operation.

    Args:
        operand1: The first number.
        operand2: The second number.
        operator: The operator string (+, -, *, /).

    Returns:
        The result of the operation.

    Raises:
        ValueError: If division by zero or unsupported operator.
    """
    if operator == '+':
        return operand1 + operand2
    elif operator == '-':
        return operand1 - operand2
    elif operator == '*':
        return operand1 * operand2
    elif operator == '/':
        if operand2 == 0:
            raise ValueError("Division by zero")
        return operand1 / operand2
    else:
        raise ValueError(f"Unsupported operator: {operator}")

def evaluate_expression(tokens: list) -> float:
    """
    Evaluates a list of tokens (numbers, operators, parentheses) representing
    an infix expression, respecting operator precedence.

    Args:
        tokens: A list where elements are numbers (int/float), or strings
                for operators ('+', '-', '*', '/') and parentheses ('(', ')').

    Returns:
        The result of the evaluated expression.

    Raises:
        ValueError: For malformed expressions, unsupported tokens, or issues
                    like mismatched parentheses or division by zero.
    """
    values_stack = []  # For numbers
    ops_stack = []     # For operators and parentheses
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}

    def _apply_top_op():
        # Helper to apply the top operator from ops_stack
        # to the top two values from values_stack.
        if not ops_stack:
            # This case should ideally not be hit if logic is correct,
            # but serves as a safeguard.
            raise ValueError("Operator stack is empty when trying to apply operation.")
        op = ops_stack.pop()
        if op == '(' : # Should not happen if parentheses are matched
             raise ValueError("Mismatched parentheses: Unexpected '(' on operator stack during apply.")
        if len(values_stack) < 2:
            raise ValueError(f"Value stack does not have enough operands for operator '{op}'.")
        val2 = values_stack.pop()
        val1 = values_stack.pop()
        values_stack.append(apply_operator(val1, val2, op))

    for token in tokens:
        if isinstance(token, (int, float)):
            values_stack.append(float(token)) # Ensure floats for division
        elif token == '(':
            ops_stack.append(token)
        elif token == ')':
            while ops_stack and ops_stack[-1] != '(':
                _apply_top_op()
            if not ops_stack or ops_stack[-1] != '(':
                raise ValueError("Mismatched parentheses: missing '(' or unbalanced expression.")
            ops_stack.pop()  # Pop the '('
        elif token in precedence: # Token is an operator
            while (ops_stack and ops_stack[-1] != '(' and
                   precedence.get(ops_stack[-1], 0) >= precedence.get(token, 0)):
                _apply_top_op()
            ops_stack.append(token)
        else:
            raise ValueError(f"Unsupported token: {token}")

    # After all tokens are processed, apply remaining operators
    while ops_stack:
        if ops_stack[-1] == '(': # Mismatched parentheses
            raise ValueError("Mismatched parentheses: extra '(' at end of expression.")
        _apply_top_op()

    if len(values_stack) == 1 and not ops_stack:
        return values_stack[0]
    elif not values_stack and not ops_stack and not tokens:
        raise ValueError("Cannot evaluate an empty expression.")
    else:
        # Catches cases like "1 2" (too many values) or other structural issues.
        raise ValueError("Invalid expression format or insufficient operands/operators at the end.")

if __name__ == '__main__':
    # Test cases for is_valid_expression
    print("--- Testing is_valid_expression ---")
    print(f"'1 + 1': {is_valid_expression('1 + 1')}")  # Expected: True
    print(f"'2 * (3 - 1)': {is_valid_expression('2 * (3 - 1)')}")  # Expected: True
    print(f"'10 / 2': {is_valid_expression('10 / 2')}")  # Expected: True
    print(f"'  ( 5 )  ': {is_valid_expression('  ( 5 )  ')}") # Expected: True
    print(f"'1+1': {is_valid_expression('1+1')}") # Expected: True
    print(f"'-5 + (3*2)': {is_valid_expression('-5 + (3*2)')}") # Expected: True (unary minus is fine as it's part of allowed chars)

    print(f"'1 + 1a': {is_valid_expression('1 + 1a')}")  # Expected: False (contains 'a')
    print(f"'import os': {is_valid_expression('import os')}")  # Expected: False (contains letters)
    print(f"'1 + 1; print()': {is_valid_expression('1 + 1; print()')}")  # Expected: False (contains ';')
    eval_test_str = 'eval("1+1")'
    print(f"'eval(\"1+1\")': {is_valid_expression(eval_test_str)}") # Expected: False (contains letters and quotes)
    print(f"'1 % 2': {is_valid_expression('1 % 2')}") # Expected: False (contains '%')
    print(f"Empty string '': {is_valid_expression('')}") # Expected: True (empty string matches zero occurrences)
    print(f"Only spaces '   ': {is_valid_expression('   ')}") # Expected: True

    # Test cases for evaluate_expression
    print("\n--- Testing evaluate_expression ---")
    print("--- Correct expressions ---")
    test_expressions_correct = {
        "1 + 1": ([1, '+', 1], 2.0),
        "2 * 3 - 1": ([2, '*', 3, '-', 1], 5.0),
        "10 / 2": ([10, '/', 2], 5.0),
        "2 * (3 + 1)": ([2, '*', '(', 3, '+', 1, ')'], 8.0),
        "1 + 2 * 3": ([1, '+', 2, '*', 3], 7.0),
        "(1 + 2) * 3": (['(', 1, '+', 2, ')', '*', 3], 9.0),
        "8 / 2 * (1 + 1)": ([8, '/', 2, '*', '(', 1, '+', 1, ')'], 8.0),
        "10": ([10], 10.0),
        "5 - 3 + 2": ([5, '-', 3, '+', 2], 4.0),
        "10 / 2 * 3": ([10, '/', 2, '*', 3], 15.0),
        "((1 + 1) * 2) / 4": (['(', '(', 1, '+', 1, ')', '*', 2, ')', '/', 4], 1.0),
        "3 * (2 + (4 - 1)) / 5": ([3, '*', '(', 2, '+', '(', 4, '-', 1, ')', ')', '/', 5], 3.0) # 3 * (2+3)/5 = 3*5/5 = 3
    }

    for expr_str, (tokens, expected) in test_expressions_correct.items():
        try:
            # Pass a copy of tokens if tokens list could be modified by the function,
            # though current evaluate_expression does not modify input list.
            result = evaluate_expression(list(tokens))
            # Using round for float comparisons to handle potential precision issues
            is_pass = abs(result - expected) < 1e-9
            print(f"Expression: {expr_str} (Tokens: {tokens}) -> Result: {result}, Expected: {expected} -> {'Pass' if is_pass else 'Fail'}")
        except ValueError as e:
            print(f"Expression: {expr_str} (Tokens: {tokens}) -> Error: {e}, Expected: {expected} -> Fail (unexpected error)")
        except Exception as e:
            print(f"Expression: {expr_str} (Tokens: {tokens}) -> Unexpected Exception: {e}, Expected: {expected} -> Fail")


    print("\n--- Error-raising expressions ---")
    test_expressions_error = {
        "1 / 0": ([1, '/', 0], "Division by zero"),
        "1 % 2": ([1, '%', 2], "Unsupported token: %"),
        "(1 + 2": (['(', 1, '+', 2], "Mismatched parentheses: extra '(' at end of expression."),
        "1 + )": ([1, '+', ')'], "Value stack does not have enough operands for operator '+'."), # Error from _apply_top_op
        "1 +": ([1, '+'], "Value stack does not have enough operands for operator '+'."),
        "empty": ([], "Cannot evaluate an empty expression."),
        "operator only *": (['*'], "Value stack does not have enough operands for operator '*'."),
        "1 2": ([1, 2], "Invalid expression format or insufficient operands/operators at the end."),
        "1 + * 2": ([1, '+', '*', 2], "Value stack does not have enough operands for operator '*'."), # '+' pushed, then '*' tries to apply, needs 2 operands for '*'
        "())": ([ '(', ')', ')'], "Mismatched parentheses: missing '(' or unbalanced expression."), # First () pops, then extra )
        "(()": ([ '(', '(', ')'], "Mismatched parentheses: extra '(' at end of expression."),
    }

    for expr_str, (tokens, expected_error_part) in test_expressions_error.items():
        try:
            result = evaluate_expression(list(tokens))
            print(f"Expression: {expr_str} (Tokens: {tokens}) -> Result: {result}, Expected error containing: '{expected_error_part}' -> Fail (no error or wrong result)")
        except ValueError as e:
            if expected_error_part in str(e):
                print(f"Expression: {expr_str} (Tokens: {tokens}) -> Error: '{e}', Expected error containing: '{expected_error_part}' -> Pass")
            else:
                print(f"Expression: {expr_str} (Tokens: {tokens}) -> Error: '{e}', Expected error containing: '{expected_error_part}' -> Fail (wrong error message)")
        except Exception as e:
            print(f"Expression: {expr_str} (Tokens: {tokens}) -> Unexpected Exception: {type(e).__name__}: {e}, Expected error containing: '{expected_error_part}' -> Fail")
