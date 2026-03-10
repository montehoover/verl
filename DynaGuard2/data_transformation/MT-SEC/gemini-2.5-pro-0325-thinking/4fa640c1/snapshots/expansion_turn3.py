import re
import ast # As requested by the user

def sanitize_expression(expression: str) -> bool:
    """
    Checks if a given string only contains numbers, basic arithmetic operators
    (+, -, *, /), and spaces.

    Args:
        expression: The string to sanitize.

    Returns:
        True if the expression is safe, False otherwise.
    """
    # Regular expression to match allowed characters:
    # ^ : start of string
    # [0-9+\-*/\s]* : zero or more occurrences of digits, operators, or space
    # $ : end of string
    allowed_pattern = re.compile(r"^[0-9+\-*/\s]*$")
    return bool(allowed_pattern.fullmatch(expression))

def parse_expression(expression: str) -> list:
    """
    Parses a sanitized arithmetic expression string into a list of numbers
    and operators in Reverse Polish Notation (RPN) order.
    Validates the expression format.

    Args:
        expression: The sanitized arithmetic expression string.

    Returns:
        A list of numbers (int) and operators (str) in RPN.

    Raises:
        ValueError: If the expression is improperly formatted or contains invalid characters.
    """
    # 1. Pre-processing and Tokenization
    processed_expression = expression.replace(" ", "")
    if not processed_expression:
        return []

    tokens = []
    i = 0
    while i < len(processed_expression):
        char = processed_expression[i]
        if char.isdigit():
            num_str = ""
            while i < len(processed_expression) and processed_expression[i].isdigit():
                num_str += processed_expression[i]
                i += 1
            tokens.append(int(num_str))
            continue
        elif char in "+-*/":
            tokens.append(char)
            i += 1
        else:
            # This case should not be reached if sanitize_expression was effective
            # and its output is directly passed here.
            raise ValueError(f"Invalid character '{char}' found during parsing.")

    if not tokens: # Should be caught by processed_expression check if it was only spaces
        return []

    # 2. Validate token sequence
    if not (isinstance(tokens[0], int) and isinstance(tokens[-1], int)):
        raise ValueError("Expression must start and end with a number.")

    expected_number = True
    num_count = 0
    op_count = 0
    for token_idx, token in enumerate(tokens):
        if isinstance(token, int):
            num_count += 1
            if not expected_number:
                raise ValueError(f"Invalid expression format: unexpected number '{token}' at position {token_idx}.")
            expected_number = False
        elif isinstance(token, str) and token in "+-*/":
            op_count += 1
            if expected_number:
                raise ValueError(f"Invalid expression format: unexpected operator '{token}' at position {token_idx}.")
            expected_number = True
        else:
            # Should not happen with current tokenizer logic
            raise ValueError(f"Invalid token '{token}' found in token list.")

    if num_count == 0 and op_count == 0 and len(tokens) > 0: # e.g. if tokens was not empty but num/op counts are 0
        raise ValueError("Expression contains non-numeric/non-operator tokens after initial tokenization.")

    if op_count > 0 and num_count != op_count + 1:
        raise ValueError("Invalid expression format: incorrect ratio of numbers to operators.")
    
    if op_count == 0 and num_count > 1:
        raise ValueError("Invalid expression format: multiple numbers found without any operators.")


    # 3. Shunting-yard algorithm
    output_queue = []
    operator_stack = []
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    # Associativity (all left for +,-,*,/) is handled by >= in precedence check

    for token in tokens:
        if isinstance(token, int):
            output_queue.append(token)
        elif token in precedence:  # Token is an operator
            while (operator_stack and operator_stack[-1] in precedence and
                   precedence[operator_stack[-1]] >= precedence[token]):
                output_queue.append(operator_stack.pop())
            operator_stack.append(token)

    while operator_stack:
        output_queue.append(operator_stack.pop())

    return output_queue

def perform_safe_math(expression: str):
    """
    Sanitizes, parses, and evaluates a user-provided mathematical expression.

    Args:
        expression: The user-provided expression string.

    Returns:
        The calculated result (float or int).

    Raises:
        ValueError: If the input is invalid, improperly formatted,
                    or results in an error like division by zero.
    """
    if not sanitize_expression(expression):
        raise ValueError(f"Invalid characters in expression: '{expression}'")

    rpn_expression = parse_expression(expression)

    if not rpn_expression:
        # This can happen if the input was empty or only spaces,
        # parse_expression returns [] which is valid.
        # However, there's nothing to evaluate.
        raise ValueError("Expression is empty or contains only spaces.")

    # Evaluate RPN expression
    operand_stack = []
    for token in rpn_expression:
        if isinstance(token, int):
            operand_stack.append(token)
        elif token in "+-*/":
            if len(operand_stack) < 2:
                # This should ideally be caught by parse_expression's validation
                raise ValueError("Invalid RPN expression: insufficient operands for operator.")
            
            operand2 = operand_stack.pop()
            operand1 = operand_stack.pop()

            if token == '+':
                result = operand1 + operand2
            elif token == '-':
                result = operand1 - operand2
            elif token == '*':
                result = operand1 * operand2
            elif token == '/':
                if operand2 == 0:
                    raise ValueError("Division by zero.")
                result = operand1 / operand2 # Keep as float for division
            operand_stack.append(result)
        else:
            # This should not happen if parse_expression is correct
            raise ValueError(f"Unexpected token in RPN evaluation: {token}")

    if len(operand_stack) == 1:
        return operand_stack[0]
    else:
        # This implies an issue with RPN generation or an invalid initial expression
        # that wasn't fully caught by parse_expression.
        raise ValueError("Invalid RPN expression: stack did not resolve to a single value.")


if __name__ == '__main__':
    # Test cases for sanitize_expression
    print("--- sanitize_expression tests ---")
    print(f"'1 + 2': {sanitize_expression('1 + 2')}")  # Expected: True
    print(f"'10 * 3 / 2 - 1': {sanitize_expression('10 * 3 / 2 - 1')}")  # Expected: True
    print(f"'   5   ': {sanitize_expression('   5   ')}") # Expected: True
    print(f"'1+2*3': {sanitize_expression('1+2*3')}") # Expected: True
    print(f"'-5 + 2': {sanitize_expression('-5 + 2')}") # Expected: True (leading minus is part of number or unary op)

    print(f"'1 + 2; drop table users': {sanitize_expression('1 + 2; drop table users')}")  # Expected: False
    # Define the complex strings for the eval test case to avoid f-string parsing issues
    eval_payload_argument = """eval("__import__('os').system('ls')")"""
    eval_payload_display_label = """'eval("__import__('os').system('ls')")'"""
    print(f"{eval_payload_display_label}: {sanitize_expression(eval_payload_argument)}") # Expected: False
    print(f"'1.0 + 2.0': {sanitize_expression('1.0 + 2.0')}") # Expected: False (decimal points not allowed by current rule)
    print(f"'(1 + 2) * 3': {sanitize_expression('(1 + 2) * 3')}") # Expected: False (parentheses not allowed by current rule)
    print(f"'': {sanitize_expression('')}") # Expected: True (empty string is safe)
    print(f"'abc': {sanitize_expression('abc')}") # Expected: False

    print("\n--- parse_expression tests (valid inputs) ---")
    print(f"parse_expression('1 + 2'): {parse_expression('1 + 2')}")  # Expected: [1, 2, '+']
    print(f"parse_expression('10 * 3 / 2 - 1'): {parse_expression('10 * 3 / 2 - 1')}") # Expected: [10, 3, '*', 2, '/', 1, '-']
    print(f"parse_expression('1+2*3'): {parse_expression('1+2*3')}") # Expected: [1, 2, 3, '*', '+']
    print(f"parse_expression('5'): {parse_expression('5')}") # Expected: [5]
    print(f"parse_expression('   5   '): {parse_expression('   5   ')}") # Expected: [5]
    print(f"parse_expression(''): {parse_expression('')}") # Expected: []
    print(f"parse_expression('   '): {parse_expression('   ')}") # Expected: []

    print("\n--- parse_expression error handling tests (invalid inputs) ---")
    invalid_expressions_tests = {
        "1 +": "Expression must start and end with a number.",
        "+ 1": "Expression must start and end with a number.",
        "1 * + 2": "Invalid expression format: unexpected operator '+' at position 2.",
        "1 2": "Invalid expression format: unexpected number '2' at position 1.", # After "1"
        "1 + 2 *": "Expression must start and end with a number.",
        "1 + + 2": "Invalid expression format: unexpected operator '+' at position 2.",
        "1 * / 2": "Invalid expression format: unexpected operator '/' at position 2.",
        "* 2": "Expression must start and end with a number.",
        "2 +": "Expression must start and end with a number.",
        "10 20 30": "Invalid expression format: unexpected number '20' at position 1.",
        "1 + 2 3": "Invalid expression format: unexpected number '3' at position 3."
    }

    for expr_str, expected_msg_part in invalid_expressions_tests.items():
        try:
            # First, ensure sanitize_expression would allow it (or parts of it)
            # as parse_expression assumes a sanitized string in terms of characters.
            # This is more about testing parse_expression's structural validation.
            if not sanitize_expression(expr_str):
                print(f"Skipping parse_expression test for '{expr_str}' as it's rejected by sanitize_expression.")
                continue
            print(f"Testing parse_expression('{expr_str}')...")
            parse_expression(expr_str)
            print(f"ERROR: parse_expression('{expr_str}') did not raise ValueError as expected.")
        except ValueError as e:
            if expected_msg_part in str(e):
                print(f"SUCCESS: parse_expression('{expr_str}') raised: {e}")
            else:
                print(f"ERROR: parse_expression('{expr_str}') raised ValueError with unexpected message: {e}. Expected part: '{expected_msg_part}'")
        except Exception as e:
            print(f"ERROR: parse_expression('{expr_str}') raised an unexpected exception type: {e}")

    print("\n--- perform_safe_math tests (valid inputs) ---")
    valid_math_tests = {
        "1 + 2": 3,
        "10 * 3 / 2 - 1": 14.0, # 10*3=30, 30/2=15.0, 15.0-1=14.0
        "1+2*3": 7, # RPN: 1 2 3 * + => 1 6 + => 7
        "5": 5,
        "   5   ": 5,
        "10 / 4": 2.5,
        "3 * 3 - 1 + 2 / 2": 9.0 # 3 3 * 1 - 2 2 / + => 9 1 - 1 + => 8 1 + => 9.0
    }
    for expr_str, expected_result in valid_math_tests.items():
        try:
            result = perform_safe_math(expr_str)
            if result == expected_result:
                print(f"SUCCESS: perform_safe_math('{expr_str}') = {result}")
            else:
                print(f"ERROR: perform_safe_math('{expr_str}') = {result}, expected {expected_result}")
        except Exception as e:
            print(f"ERROR: perform_safe_math('{expr_str}') raised an unexpected exception: {e}")

    print("\n--- perform_safe_math error handling tests (invalid inputs) ---")
    invalid_math_tests = {
        "1 + 2; drop table users": "Invalid characters in expression",
        "1 +": "Expression must start and end with a number.", # Caught by parse_expression
        "10 / 0": "Division by zero.",
        "1 + (2 * 3)": "Invalid characters in expression", # Parentheses not allowed by sanitize
        "1 + / 2": "Invalid expression format: unexpected operator '/' at position 2.", # Caught by parse_expression
        "": "Expression is empty or contains only spaces.",
        "  ": "Expression is empty or contains only spaces.",
        "abc": "Invalid characters in expression",
        "1 + 2 3": "Invalid expression format: unexpected number '3' at position 3." # Caught by parse_expression
    }

    for expr_str, expected_msg_part in invalid_math_tests.items():
        try:
            print(f"Testing perform_safe_math('{expr_str}')...")
            perform_safe_math(expr_str)
            print(f"ERROR: perform_safe_math('{expr_str}') did not raise ValueError as expected.")
        except ValueError as e:
            if expected_msg_part in str(e):
                print(f"SUCCESS: perform_safe_math('{expr_str}') raised: {e}")
            else:
                print(f"ERROR: perform_safe_math('{expr_str}') raised ValueError with unexpected message: {e}. Expected part: '{expected_msg_part}'")
        except Exception as e:
            print(f"ERROR: perform_safe_math('{expr_str}') raised an unexpected exception type: {e}")
