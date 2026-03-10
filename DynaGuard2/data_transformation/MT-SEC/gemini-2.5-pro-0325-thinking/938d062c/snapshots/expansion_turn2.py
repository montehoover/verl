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
    # Regex to allow numbers, operators (+, -, *, /), parentheses, and spaces.
    # ^ : asserts position at start of the string.
    # [0-9+\-*/()\s] : matches any character in the set:
    #   0-9 : digits
    #   +   : literal plus
    #   -   : literal minus
    #   *   : literal asterisk
    #   /   : literal slash
    #   ()  : literal parentheses (no need to escape inside [])
    #   \s  : whitespace characters
    # + : matches the previous token between one and unlimited times.
    # $ : asserts position at the end of the string.
    # Using r"" for raw string to handle backslashes correctly if they were needed for special chars.
    # For this specific pattern, it's not strictly necessary but good practice.
    pattern = r"^[0-9+\-*/()\s]+$"
    if re.fullmatch(pattern, expression):
        return True
    return False

def parse_expression(expression: str) -> list:
    """
    Parses a mathematical expression string into Reverse Polish Notation (RPN).
    Assumes the expression has been validated by is_valid_expression regarding
    allowed characters. This function handles structural parsing like operator
    precedence and parentheses matching.

    Args:
        expression: The mathematical expression string.

    Returns:
        A list representing the expression in RPN.
        Numbers are integers, operators are strings.

    Raises:
        ValueError: If parentheses are mismatched or other structural issues occur.
    """
    # Remove all whitespace from the expression for simpler tokenization
    processed_expression = expression.replace(" ", "")

    # Tokenize the expression.
    # \d+ matches one or more digits (integers).
    # [+\-*/()] matches one character from the set of operators and parentheses.
    tokens = re.findall(r"\d+|[+\-*/()]", processed_expression)

    output_queue = []
    operator_stack = []
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    # All supported operators (+, -, *, /) are left-associative.

    for token in tokens:
        if token.isdigit():
            output_queue.append(int(token))
        elif token == '(':
            operator_stack.append(token)
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                output_queue.append(operator_stack.pop())
            # If stack is empty or top is not '(', then parentheses are mismatched
            if not operator_stack or operator_stack[-1] != '(':
                raise ValueError("Mismatched parentheses: missing '(' or misplaced ')'")
            operator_stack.pop()  # Pop the '(' from the stack and discard it
        else:  # Token is an operator
            # While stack is not empty, top is not '(',
            # and top operator has greater or equal precedence (for left-associativity)
            while (operator_stack and
                   operator_stack[-1] != '(' and
                   precedence.get(operator_stack[-1], 0) >= precedence.get(token, 0)):
                output_queue.append(operator_stack.pop())
            operator_stack.append(token)

    # After processing all tokens, pop any remaining operators from the stack to the output
    while operator_stack:
        top_op = operator_stack.pop()
        if top_op == '(':
            # If a '(' is found here, it means it was unclosed
            raise ValueError("Mismatched parentheses: unclosed '('")
        output_queue.append(top_op)

    return output_queue

if __name__ == '__main__':
    # Test cases
    valid_expressions = [
        "1 + 1",
        "2 * (3 - 1)",
        "10 / 2",
        "   5   ",
        "12345",
        "(5 * (3+2))/(8-3)"
    ]
    invalid_expressions = [
        "1 + a",
        "eval('__import__(\"os\").system(\"echo unsafe\")')",
        "1 + 1;",
        "print('hello')",
        "1 & 2",
        "import os"
    ]

    print("Testing valid expressions:")
    for expr in valid_expressions:
        print(f"'{expr}': is_valid_expression -> {is_valid_expression(expr)}")

    print("\nTesting invalid expressions:")
    for expr in invalid_expressions:
        print(f"'{expr}': is_valid_expression -> {is_valid_expression(expr)}")

    print("\nTesting parse_expression (with valid expressions):")
    parse_test_expressions = [
        "1 + 1",
        "2 * (3 - 1)",
        "10 / 2",
        "3 + 4 * 2 / ( 1 - 0 )", # Using 0 is fine for parsing
        " ( 1 + 2 ) * 3 - 4 / 2 ",
        "100",
        "5 * 2 + 3 * 4", # Expected: 5 2 * 3 4 * +
        "1 + 2 - 3 + 4"  # Expected: 1 2 + 3 - 4 + (left-associativity)
    ]

    for expr_str in parse_test_expressions:
        print(f"Input: '{expr_str}'")
        if is_valid_expression(expr_str):
            try:
                rpn = parse_expression(expr_str)
                print(f"  RPN: {rpn}")
            except ValueError as e:
                print(f"  Error during parsing: {e}")
        else:
            # This path should ideally not be taken for these test cases
            print(f"  Skipped parsing as it's invalid per is_valid_expression: '{expr_str}'")
    
    print("\nTesting parse_expression (with mismatched parentheses or structural issues):")
    # Note: is_valid_expression checks characters, parse_expression checks structure.
    # Some of these might pass is_valid_expression but fail parse_expression.
    error_test_expressions = [
        "(1 + 2",       # Unclosed parenthesis
        "1 + 2)",       # Unopened parenthesis
        "((1+2)*3",     # Unclosed parenthesis
        "(1+)2)*3",     # Passes is_valid_expression, but +) is not standard.
                        # Tokenizer: '(', '1', '+', ')', '2', ')', '*', '3'
                        # parse_expression should error on the second ')'
        "1 * (2 + )"    # Passes is_valid_expression. Tokenizer: 1 * ( 2 + )
                        # parse_expression might error or produce unexpected RPN
                        # depending on how it handles trailing operators before ')'
                        # Current Shunting-yard expects an operand after an operator unless it's end of expression.
                        # This specific case: '1', '*', '(', '2', '+', ')'
                        # When ')' is met, '+' is popped. RPN: [1, 2, '+', '*']
                        # This is (1 * (2+)). If an evaluator expects two operands for '+', it would fail there.
                        # The parser itself doesn't check arity of operators.
    ]

    for expr_str in error_test_expressions:
        print(f"Input: '{expr_str}'")
        valid_chars = is_valid_expression(expr_str)
        print(f"  is_valid_expression: {valid_chars}")
        try:
            rpn = parse_expression(expr_str)
            print(f"  RPN: {rpn}")
        except ValueError as e:
            print(f"  Error during parsing: {e}")
