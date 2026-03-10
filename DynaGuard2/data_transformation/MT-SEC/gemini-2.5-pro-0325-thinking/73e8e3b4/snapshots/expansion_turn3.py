import re

def sanitize_expression(expression: str) -> bool:
    """
    Checks if a given string only contains numbers, basic arithmetic
    operators (+, -, *, /), parentheses, and spaces.

    Args:
        expression: The string to validate.

    Returns:
        True if the expression is valid, False otherwise.
    """
    # Regex to allow numbers, operators (+, -, *, /), parentheses, and spaces.
    # ^ and $ ensure the entire string matches the pattern.
    # \d matches digits.
    # \s matches whitespace.
    # \+\-\*\/ are escaped versions of the operators.
    # \(\) are escaped versions of parentheses.
    pattern = r"^[0-9\+\-\*\/\(\)\s]*$"
    if re.fullmatch(pattern, expression):
        return True
    return False

def parse_expression(expression: str) -> list:
    """
    Parses a valid expression string into a list of numbers and operators
    in Reverse Polish Notation (RPN), respecting operator precedence.

    Args:
        expression: The sanitized expression string.

    Returns:
        A list representing the expression in RPN.
        Numbers are converted to float.
    """
    if not sanitize_expression(expression):
        raise ValueError("Invalid characters in expression")

    # Operator precedence and associativity
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    # All supported operators are left-associative

    output_queue = []
    operator_stack = []

    # Tokenize the expression:
    # This regex finds numbers (integers or floats), operators (+, -, *, /), and parentheses.
    # It also removes spaces by not capturing them explicitly.
    tokens = re.findall(r"(\d+\.?\d*|\.\d+|[+\-*/()]|\S)", expression.replace(" ", ""))
    
    # Filter out any empty strings that might result from replace and findall
    tokens = [token for token in tokens if token]


    for token in tokens:
        if token.isdigit() or (token.startswith('.') and token[1:].isdigit()) or ('.' in token and token.replace('.', '', 1).isdigit()):
            # Token is a number
            output_queue.append(float(token))
        elif token in precedence:
            # Token is an operator
            while (operator_stack and operator_stack[-1] in precedence and
                   precedence[operator_stack[-1]] >= precedence[token]):
                output_queue.append(operator_stack.pop())
            operator_stack.append(token)
        elif token == '(':
            operator_stack.append(token)
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                output_queue.append(operator_stack.pop())
            if not operator_stack or operator_stack[-1] != '(':
                raise ValueError("Mismatched parentheses")
            operator_stack.pop()  # Pop the '('
        else:
            # This case should ideally not be reached if sanitize_expression works correctly
            # and tokenization is robust for valid expressions.
            # However, as a safeguard:
            raise ValueError(f"Unknown token: {token}")


    while operator_stack:
        operator = operator_stack.pop()
        if operator == '(' or operator == ')':
            raise ValueError("Mismatched parentheses")
        output_queue.append(operator)

    return output_queue

def evaluate_rpn(rpn_tokens: list) -> float:
    """
    Evaluates a list of tokens in Reverse Polish Notation.

    Args:
        rpn_tokens: A list of numbers (float) and operators (str).

    Returns:
        The result of the calculation.

    Raises:
        ValueError: If the RPN expression is invalid, encounters division by zero,
                    or has insufficient operands.
    """
    stack = []
    operators = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: a / b if b != 0 else (_ for _ in ()).throw(ValueError("Division by zero"))
    }

    if not rpn_tokens:
        raise ValueError("Cannot evaluate an empty expression.")

    for token in rpn_tokens:
        if isinstance(token, float):
            stack.append(token)
        elif token in operators:
            if len(stack) < 2:
                raise ValueError("Invalid expression: insufficient operands for operator.")
            operand2 = stack.pop()
            operand1 = stack.pop()
            result = operators[token](operand1, operand2)
            stack.append(result)
        else:
            # This should not happen if parse_expression is correct
            raise ValueError(f"Unknown token in RPN list: {token}")

    if len(stack) == 1:
        return stack[0]
    else:
        raise ValueError("Invalid expression: stack should contain one item after evaluation.")

def evaluate_and_compute(expression: str) -> float:
    """
    Validates, parses, and computes the result of a mathematical expression string.

    Args:
        expression: The mathematical expression string.

    Returns:
        The evaluated result as a float.

    Raises:
        ValueError: For invalid characters, syntax errors (e.g., mismatched parentheses),
                    division by zero, or other evaluation errors.
    """
    if not sanitize_expression(expression):
        raise ValueError("Invalid characters in expression.")
    
    # Handle empty or whitespace-only strings after sanitization
    if not expression.strip():
        raise ValueError("Cannot evaluate an empty expression.")

    rpn_tokens = parse_expression(expression)
    
    # parse_expression might return an empty list for an empty (but sanitized) string,
    # which evaluate_rpn now handles.
    # However, parse_expression itself should ideally raise error or be handled if it returns empty for non-empty input.
    # Current parse_expression returns [] for "" or "   ".
    # The check for expression.strip() above handles this before parse_expression.

    return evaluate_rpn(rpn_tokens)

if __name__ == '__main__':
    # Test cases
    valid_expressions = [
        "1 + 2",
        " (3 * 4) - 5 / 6 ",
        "12345",
        "((()))",
        "1+1",
        "",
        "   "
    ]
    invalid_expressions = [
        "1 + 2a",
        "import os",
        "eval('1+1')",
        "1 & 2",
        "1 % 2",
        "print('hello')"
    ]

    print("Testing valid expressions:")
    for expr in valid_expressions:
        print(f"'{expr}': {sanitize_expression(expr)}")

    print("\nTesting invalid expressions:")
    for expr in invalid_expressions:
        print(f"'{expr}': {sanitize_expression(expr)}")

    print("\nTesting parse_expression:")
    expressions_to_parse = [
        "1 + 2",
        "3 * 4 + 2",
        "3 + 4 * 2",
        " ( 1 + 2 ) * 3 ",
        "10 / 2 * 5", # Should be (10/2)*5 = 25
        "10 / ( 2 * 5 )", # Should be 10/10 = 1
        "3.14 * 2 + 0.5",
        "1+2*3-4/2" # 1 + 6 - 2 = 5  => [1, 2, 3, '*', '+', 4, 2, '/', '-']
    ]
    for expr_str in expressions_to_parse:
        try:
            if sanitize_expression(expr_str):
                parsed = parse_expression(expr_str)
                print(f"Expression: '{expr_str}' -> Parsed (RPN): {parsed}")
            else:
                print(f"Expression: '{expr_str}' -> Invalid characters, not parsed.")
        except ValueError as e:
            print(f"Error parsing '{expr_str}': {e}")

    print("\nTesting parse_expression with invalid syntax (parentheses):")
    invalid_syntax_expressions = [
        "(1 + 2",
        "1 + 2)",
        "((1+2) * 3",
    ]
    for expr_str in invalid_syntax_expressions:
        try:
            if sanitize_expression(expr_str): # It will be sanitized, but parsing will fail
                parsed = parse_expression(expr_str)
                print(f"Expression: '{expr_str}' -> Parsed (RPN): {parsed}")
            else:
                # This branch might not be hit if sanitize allows unbalanced parentheses
                # as it only checks for allowed characters.
                print(f"Expression: '{expr_str}' -> Invalid characters, not parsed.")
        except ValueError as e:
            print(f"Error parsing '{expr_str}': {e}")

    print("\nTesting evaluate_and_compute:")
    eval_test_cases = {
        "1 + 2": 3.0,
        "3 * 4 + 2": 14.0,
        "3 + 4 * 2": 11.0,
        "(1 + 2) * 3": 9.0,
        "10 / 2 * 5": 25.0,
        "10 / (2 * 5)": 1.0,
        "3.14 * 2 + 0.5": 6.78,
        "1+2*3-4/2": 5.0,
        "100": 100.0,
        " ( 5 ) ": 5.0
    }
    for expr_str, expected_result in eval_test_cases.items():
        try:
            result = evaluate_and_compute(expr_str)
            print(f"Expression: '{expr_str}' -> Result: {result} (Expected: {expected_result}) {'PASS' if abs(result - expected_result) < 1e-9 else 'FAIL'}")
        except ValueError as e:
            print(f"Error evaluating '{expr_str}': {e} (Expected: {expected_result}) FAIL")

    print("\nTesting evaluate_and_compute with errors:")
    eval_error_cases = [
        "1 + 2a",          # Invalid character
        "(1 + 2",          # Mismatched parentheses (parsing error)
        "1 +",             # Insufficient operands (RPN eval error)
        "1 / 0",           # Division by zero
        "1 2 +",           # Invalid format for this parser (space separated RPN-like)
        "",                # Empty expression
        "   ",             # Whitespace only expression
        "1 * (2 + )",      # Syntax error within parentheses
        "import os"        # Unsafe
    ]
    for expr_str in eval_error_cases:
        try:
            result = evaluate_and_compute(expr_str)
            print(f"Error evaluating '{expr_str}': Expected error, but got {result} FAIL")
        except ValueError as e:
            print(f"Error evaluating '{expr_str}': {e} PASS")
