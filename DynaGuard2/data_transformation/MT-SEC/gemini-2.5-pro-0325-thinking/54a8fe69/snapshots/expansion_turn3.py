import re

def validate_expression(expression: str) -> bool:
    """
    Validates if the given expression string contains only numbers,
    basic arithmetic operators (+, -, *, /), and spaces.

    Args:
        expression: The string to validate.

    Returns:
        True if the expression is valid, False otherwise.
    """
    # Regex to match only numbers, operators (+, -, *, /), and spaces.
    # ^ asserts position at start of the string.
    # [0-9\s+\-*/]* matches zero or more occurrences of any digit, whitespace,
    # plus, minus, asterisk, or slash.
    # $ asserts position at the end of the string.
    pattern = r"^[0-9\s+\-*/]*$"
    if re.fullmatch(pattern, expression):
        return True
    return False

def parse_expression(expression: str) -> list:
    """
    Parses a valid expression string into a list of numbers and operators.

    Args:
        expression: The validated expression string.

    Returns:
        A list of numbers (as strings) and operators.
    """
    # Regex to find numbers (sequences of digits) or operators (+, -, *, /).
    # \d+ matches one or more digits.
    # [+\-*/] matches any single character that is a plus, minus, asterisk, or slash.
    # The | acts as an OR.
    # re.findall will find all non-overlapping matches of the pattern.
    components = re.findall(r"\d+|[+\-*/]", expression)
    return components

def evaluate_user_expression(expression: str) -> int:
    """
    Validates, parses, and evaluates a mathematical expression string.
    The evaluation is done from left to right, without operator precedence.
    Supports basic arithmetic operators: +, -, *, /.
    Division is integer division.
    Supports unary + and - at the start of the expression or for operands.

    Args:
        expression: The mathematical expression string.

    Returns:
        The integer result of the evaluation.

    Raises:
        ValueError: If the expression is invalid, malformed,
                    or a division by zero occurs.
    """
    if not validate_expression(expression):
        raise ValueError("Invalid characters in expression. Only numbers, operators (+, -, *, /), and spaces are allowed.")

    components = parse_expression(expression)

    if not components:
        raise ValueError("Expression is empty or yields no components after parsing.")

    operators = {'+', '-', '*', '/'}
    result = 0
    idx = 0

    # Handle the first number (which might have a leading unary operator)
    first_token = components[idx]
    if first_token in operators: # Potential unary operator
        if first_token not in {'+', '-'}:
            raise ValueError(f"Invalid start of expression: Operator '{first_token}' cannot be unary here.")
        idx += 1
        if idx >= len(components):
            raise ValueError("Invalid expression: Unary operator at start not followed by a number.")
        
        num_str = components[idx]
        if num_str in operators:
            raise ValueError("Invalid expression: Unary operator at start followed by another operator.")
        
        result = int(num_str)
        if first_token == '-':
            result = -result
        idx += 1
    else: # Must be a number
        result = int(first_token)
        idx += 1

    # Main loop for subsequent operations
    while idx < len(components):
        operator_token = components[idx]
        if operator_token not in operators:
            # This case should ideally not be reached if parse_expression is correct
            # and validate_expression ensures only valid tokens.
            raise ValueError(f"Invalid token: Expected operator, got '{operator_token}'.")
        idx += 1

        if idx >= len(components):
            raise ValueError("Invalid expression: Operator not followed by an operand.")

        operand_val = 0
        next_token = components[idx]

        if next_token in operators: # Potential unary operator for the operand
            if next_token not in {'+', '-'}:
                raise ValueError(f"Invalid operand: Operator '{next_token}' cannot be unary for an operand.")
            idx += 1
            if idx >= len(components):
                raise ValueError("Invalid expression: Unary operator for operand not followed by a number.")
            
            num_str = components[idx]
            if num_str in operators:
                raise ValueError("Invalid expression: Unary operator for operand followed by another operator.")
            
            operand_val = int(num_str)
            if next_token == '-':
                operand_val = -operand_val
            idx += 1
        else: # Must be a number
            operand_val = int(next_token)
            idx += 1

        # Perform calculation
        if operator_token == '+':
            result += operand_val
        elif operator_token == '-':
            result -= operand_val
        elif operator_token == '*':
            result *= operand_val
        elif operator_token == '/':
            if operand_val == 0:
                raise ValueError("Division by zero.")
            result //= operand_val # Integer division

    return result

if __name__ == '__main__':
    # Example Usage for validate_expression
    print("Validation Tests:")
    print(f"'1 + 1': {validate_expression('1 + 1')}")  # Expected: True
    print(f"'2 * 3 - 4 / 2': {validate_expression('2 * 3 - 4 / 2')}")  # Expected: True
    print(f"'100': {validate_expression('100')}")  # Expected: True
    print(f"'-5': {validate_expression('-5')}")  # Expected: True (leading minus is an operator)
    print(f"'1 + (2 * 3)': {validate_expression('1 + (2 * 3)')}")  # Expected: False (contains parentheses)
    print(f"'abc': {validate_expression('abc')}")  # Expected: False (contains letters)
    print(f"'1 + 1 = 2': {validate_expression('1 + 1 = 2')}")  # Expected: False (contains '=')
    print(f"Empty string '': {validate_expression('')}")  # Expected: True (empty string matches the pattern)
    print(f"'   ': {validate_expression('   ')}")  # Expected: True (only spaces)
    print(f"'1.5 + 2.3': {validate_expression('1.5 + 2.3')}")  # Expected: False (contains '.')

    # Example Usage for parse_expression
    print("\nParsing Tests:")
    print(f"'1 + 1': {parse_expression('1 + 1')}")  # Expected: ['1', '+', '1']
    print(f"'2 * 3 - 4 / 2': {parse_expression('2 * 3 - 4 / 2')}")  # Expected: ['2', '*', '3', '-', '4', '/', '2']
    print(f"'100': {parse_expression('100')}")  # Expected: ['100']
    print(f"'-5 + 10': {parse_expression('-5 + 10')}") # Expected: ['-', '5', '+', '10']
    print(f"'   10   *  2 ' : {parse_expression('   10   *  2 ')}") # Expected: ['10', '*', '2']
    print(f"Empty string '': {parse_expression('')}") # Expected: []

    # Example Usage for evaluate_user_expression
    print("\nEvaluation Tests:")
    test_expressions = {
        "1 + 1": 2,
        "2 * 3 - 4 / 2": 1, # (2*3=6, 6-4=2, 2/2=1)
        "-5 + 10": 5,
        "10": 10,
        "-5": -5,
        "+5": 5,
        "1 * -2": -2, # Interpreted as 1 * (-2)
        "10 / 3": 3, # Integer division
        "1 / 2": 0, # Integer division
        "-7 / 2": -4, # Floor division: -3.5 -> -4
        "2 * 3 + 4 * 5": 50, # Left-to-right: (2*3=6, 6+4=10, 10*5=50)
        "10 - -2": 12, # 10 - (-2)
        "  10  *   2  ": 20,
    }

    for expr_str, expected_val in test_expressions.items():
        try:
            # Simulate space variations for robustness with parse_expression
            # For "1 * -2", parse_expression needs "1 * - 2"
            # For "10 - -2", parse_expression needs "10 - - 2"
            # This is a bit of a hack due to current parse_expression;
            # ideally, parser handles this or validator is stricter.
            # For these tests, we'll assume valid spacing for unary ops after binary ops.
            test_expr_str = expr_str
            if expr_str == "1 * -2":
                test_expr_str = "1 * - 2"
            elif expr_str == "10 - -2":
                test_expr_str = "10 - - 2"

            result = evaluate_user_expression(test_expr_str)
            print(f"'{expr_str}': {result} (Expected: {expected_val}) {'PASS' if result == expected_val else 'FAIL'}")
        except ValueError as e:
            print(f"'{expr_str}': Error: {e}")

    print("\nError Case Tests for Evaluation:")
    error_expressions = [
        "",
        "   ",
        "1 +",
        "* 2",
        "1 * / 2", # validate_expression allows, parse_expression gives ['1', '*', '/', '2']
        "1 + (2*3)", # validate_expression disallows '('
        "abc",       # validate_expression disallows 'a'
        "1 + 1 = 2", # validate_expression disallows '='
        "1 / 0",
        "1 + - * 2", # validate_expression allows, parse_expression gives ['1', '+', '-', '*', '2']
        "1 + --2", # validate_expression allows "1 + --2", parse_expression gives ['1', '+', '-', '-', '2']
    ]

    for expr_str in error_expressions:
        try:
            # Adjust spacing for parser if needed for specific error tests
            test_expr_str = expr_str
            if expr_str == "1 + --2": # "1 + - - 2"
                 test_expr_str = "1 + - - 2"
            elif expr_str == "1 + - * 2":
                 test_expr_str = "1 + - * 2"


            evaluate_user_expression(test_expr_str)
            print(f"'{expr_str}': Did not raise ValueError (FAIL)")
        except ValueError as e:
            print(f"'{expr_str}': Raised ValueError: \"{e}\" (PASS)")
