import re
import ast

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

def _tokenize_expression_for_compute(expression: str) -> list:
    """
    Tokenizes a validated arithmetic expression string into a list of
    numbers (float) and operators/parentheses (str).
    Handles unary minuses correctly for the evaluate_expression function.
    Assumes `is_valid_expression` has already passed on the raw expression.
    """
    expression_no_spaces = expression.replace(" ", "")
    if not expression_no_spaces:
        return []

    # Regex to find numbers (positive form) or operators/parentheses
    # Number part: \d+(?:\.\d*)?  (e.g., 123, 123.)
    #              |\.\d+         (e.g., .5)
    # Operators/Parens: [()+\-*/]
    pattern = r"(\d+(?:\.\d*)?|\.\d+|[()+\-*/])"
    
    initial_tokens_str = re.findall(pattern, expression_no_spaces)
    
    final_tokens = []
    idx = 0
    while idx < len(initial_tokens_str):
        token_str = initial_tokens_str[idx]
        
        is_unary_permissible_context = (not final_tokens) or \
                                       (final_tokens[-1] in ['(', '+', '-', '*', '/'])
        
        if token_str == '-' and is_unary_permissible_context:
            if idx + 1 < len(initial_tokens_str):
                next_token_str = initial_tokens_str[idx+1]
                # Check if next_token_str is a number string (i.e., not an operator/paren)
                if next_token_str not in ['+', '-', '*', '/', '(', ')']:
                    try:
                        num_val = float("-" + next_token_str)
                        final_tokens.append(num_val)
                        idx += 2 # Consumed '-' and the number string
                        continue
                    except ValueError:
                        # Should not happen if next_token_str was a valid number string.
                        # Fall through to treat '-' as an operator if conversion fails.
                        pass 
            # If no next token, or next token is not a number, or conversion failed,
            # treat '-' as an operator token.
            final_tokens.append(token_str)
        elif token_str in ['+', '-', '*', '/', '(', ')']:
            final_tokens.append(token_str)
        else: # Assumed to be a number string (positive form)
            try:
                final_tokens.append(float(token_str))
            except ValueError:
                # This should not be reached if is_valid_expression and regex are correct.
                raise ValueError(f"Invalid number token encountered: '{token_str}' in expression '{expression}'.")
        idx += 1
        
    return final_tokens

def compute_expression(expression: str) -> float:
    """
    Validates and computes a string arithmetic expression.

    Args:
        expression: The string expression to compute.

    Returns:
        The result of the computation.

    Raises:
        ValueError: If the expression is invalid, contains unsupported characters,
                    is malformed, or results in an arithmetic error (e.g., division by zero).
    """
    if not is_valid_expression(expression):
        raise ValueError("Expression contains invalid characters.")

    tokens = _tokenize_expression_for_compute(expression)

    try:
        result = evaluate_expression(tokens)
        return result
    except ValueError as e:
        # Re-raise to provide a consistent error source or add context.
        raise ValueError(f"Error during evaluation: {str(e)}") from e

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

    # Test cases for compute_expression
    print("\n--- Testing compute_expression ---")
    compute_tests = [
        # Valid expressions
        ("1 + 1", 2.0),
        (" 2 * (30 - 4) / 2 ", 26.0), # With spaces
        ("-5 + 10", 5.0),
        ("10 - -2", 12.0), # 10 - (-2)
        ("1.5 * -2", -3.0),
        ("10 / 2 * 3", 15.0),
        ("2 * (3 + 1)", 8.0),
        ("10", 10.0),
        (".5 * 4", 2.0),
        ("5. * 2", 10.0),
        ("1 + 2 * 3 - -4 / 2", 9.0), # 1 + 6 - (-2) = 1 + 6 + 2 = 9

        # Expressions that should raise ValueError
        ("1 + a", "Expression contains invalid characters."),
        ("1 # 2", "Expression contains invalid characters."),
        ("1 + * 2", "Error during evaluation: Value stack does not have enough operands for operator '*'."), # Tokenizer: [1.0, '+', '*', 2.0]
        ("(1 + 2", "Error during evaluation: Mismatched parentheses: extra '(' at end of expression."),
        ("1 + (2 * 3", "Error during evaluation: Mismatched parentheses: extra '(' at end of expression."), # Tokenizer: [1.0, '+', '(', 2.0, '*', 3.0]
        ("1 / 0", "Error during evaluation: Division by zero"),
        ("", "Error during evaluation: Cannot evaluate an empty expression."), # is_valid is True, tokenize gives []
        ("   ", "Error during evaluation: Cannot evaluate an empty expression."), # is_valid is True, tokenize gives []
        ("1 + )", "Error during evaluation: Value stack does not have enough operands for operator '+'."), # Tokenizer: [1.0, '+', ')']
        ("1 + ", "Error during evaluation: Value stack does not have enough operands for operator '+'."), # Tokenizer: [1.0, '+']
        ("1 2", "Error during evaluation: Invalid expression format or insufficient operands/operators at the end."), # Tokenizer: [1.0, 2.0]
        ("5 * -(2+1)", "Error during evaluation: Value stack does not have enough operands for operator '-'."), # Tokenizer: [5.0, '*', '-', '(', 2.0, '+', 1.0, ')'] -> eval error
        ("5 * --2", "Error during evaluation: Value stack does not have enough operands for operator '-'."), # Tokenizer: [5.0, '*', '-', -2.0] -> this should be 5 * (-(-2)) if we supported double unary, but tokenizer makes it 5 * -(-2) which is fine.
                                                                                                            # Actually, initial tokens: ['5', '*', '-', '-', '2']
                                                                                                            # final_tokens: [5.0, '*', '-', -2.0]
                                                                                                            # This is 5 * (- (-2.0)). This should evaluate.
                                                                                                            # Let's re-check "5 * --2"
                                                                                                            # initial: ['5', '*', '-', '-', '2']
                                                                                                            # 5.0 -> final_tokens = [5.0]
                                                                                                            # *   -> final_tokens = [5.0, '*']
                                                                                                            # -   -> unary context. next is '-'. Not a number. So append '-' -> final_tokens = [5.0, '*', '-']
                                                                                                            # -   -> unary context. next is '2'. Append -2.0 -> final_tokens = [5.0, '*', '-', -2.0]
                                                                                                            # This is 5 * operator- (-2.0). This is correct.
                                                                                                            # evaluate_expression([5.0, '*', '-', -2.0])
                                                                                                            # values: [5.0] ops: ['*']
                                                                                                            # values: [5.0, -2.0] ops: ['*', '-'] (assuming '-' has higher or equal precedence than '*' if it were unary, but it's binary here)
                                                                                                            # Precedence: '*':2, '-':1. So '*' applied first.
                                                                                                            # This test case "5 * --2" might be tricky.
                                                                                                            # If it's "5 * (-(-2))", result is 10.
                                                                                                            # My tokenizer gives [5.0, '*', '-', -2.0].
                                                                                                            # evaluate_expression:
                                                                                                            # 5.0 onto values_stack
                                                                                                            # * onto ops_stack
                                                                                                            # - (binary). Precedence of ops_stack[-1]='*' (2) >= precedence of '-' (1). Apply *.
                                                                                                            #   _apply_top_op needs 2 values. Fails if values_stack is [5.0] and next token is -2.0.
                                                                                                            #   Ah, -2.0 is pushed to values_stack first.
                                                                                                            #   values_stack = [5.0, -2.0]
                                                                                                            #   ops_stack = ['*']
                                                                                                            #   current_op = '-'
                                                                                                            #   precedence['*'] >= precedence['-'] is true.
                                                                                                            #   _apply_top_op: op='*', val2=-2.0, val1=5.0. result = -10.0. values_stack = [-10.0]
                                                                                                            #   ops_stack is now empty. Push '-'. ops_stack = ['-']
                                                                                                            # End of tokens. Apply remaining ops. op='-'. Needs 2 values. values_stack = [-10.0]. Fails.
                                                                                                            # "Value stack does not have enough operands for operator '-'." This is the expected error.
        ("5 * (2+)", "Error during evaluation: Value stack does not have enough operands for operator '+'."), # Tokenizer: [5.0, '*', '(', 2.0, '+', ')'] -> eval error
    ]

    # Adjusting the "5 * --2" test based on analysis. It should indeed error out as described.
    # The error message for "5 * -(2+1)" is also "Value stack does not have enough operands for operator '-'."

    for expr_str, expected_val_or_err in compute_tests:
        try:
            result = compute_expression(expr_str)
            if isinstance(expected_val_or_err, str): # Expected an error
                print(f"Compute: '{expr_str}' -> Result: {result}, Expected error: '{expected_val_or_err}' -> Fail (no error or wrong result)")
            else: # Expected a value
                is_pass = abs(result - expected_val_or_err) < 1e-9
                print(f"Compute: '{expr_str}' -> Result: {result}, Expected: {expected_val_or_err} -> {'Pass' if is_pass else 'Fail'}")
        except ValueError as e:
            if isinstance(expected_val_or_err, str): # Expected an error
                if expected_val_or_err in str(e):
                    print(f"Compute: '{expr_str}' -> Error: '{e}', Expected error containing: '{expected_val_or_err}' -> Pass")
                else:
                    print(f"Compute: '{expr_str}' -> Error: '{e}', Expected error containing: '{expected_val_or_err}' -> Fail (wrong error message)")
            else: # Expected a value, but got error
                print(f"Compute: '{expr_str}' -> Error: {e}, Expected: {expected_val_or_err} -> Fail (unexpected error)")
        except Exception as e:
            print(f"Compute: '{expr_str}' -> Unexpected Exception: {type(e).__name__}: {e}, Expected: {expected_val_or_err} -> Fail")
