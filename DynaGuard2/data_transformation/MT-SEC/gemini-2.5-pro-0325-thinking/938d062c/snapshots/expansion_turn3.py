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

def evaluate_expression(expression: str) -> float:
    """
    Validates, parses, and evaluates a mathematical expression string.

    Args:
        expression: The mathematical expression string.

    Returns:
        The evaluated result as a float.

    Raises:
        ValueError: If the input expression is invalid, malformed,
                    or if an error occurs during evaluation (e.g., division by zero).
    """
    if not is_valid_expression(expression):
        raise ValueError(f"Invalid characters in expression: '{expression}'")

    rpn_tokens = parse_expression(expression) # Can raise ValueError for mismatched parentheses

    if not rpn_tokens: # Handle empty or whitespace-only expressions that might pass validation/parsing
        raise ValueError("Expression is empty or results in no tokens.")

    operand_stack = []
    operators = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: a / b
    }

    for token in rpn_tokens:
        if isinstance(token, int): # Or float, if parse_expression could produce floats
            operand_stack.append(float(token)) # Store as float for division
        elif token in operators:
            if len(operand_stack) < 2:
                raise ValueError(f"Insufficient operands for operator '{token}'")
            operand2 = operand_stack.pop()
            operand1 = operand_stack.pop()
            
            if token == '/' and operand2 == 0:
                raise ValueError("Division by zero")
            
            result = operators[token](operand1, operand2)
            operand_stack.append(result)
        else:
            # This case should not be reached if parse_expression is correct
            # and only returns numbers or known operators.
            raise ValueError(f"Unknown token in RPN: {token}")

    if len(operand_stack) != 1:
        # This can happen for expressions like "1 2" (RPN: [1, 2])
        # or if parse_expression produced an invalid RPN structure.
        raise ValueError("Invalid RPN expression: stack should have 1 result at the end.")

    return operand_stack[0]

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

    print("\nTesting evaluate_expression:")
    evaluation_tests = {
        "1 + 1": 2.0,
        "2 * (3 - 1)": 4.0,
        "10 / 2": 5.0,
        "3 + 4 * 2 / ( 1 - 0 )": 11.0, # 3 + 8 / 1 = 3 + 8 = 11
        " ( 1 + 2 ) * 3 - 4 / 2 ": 7.0, # 3 * 3 - 2 = 9 - 2 = 7
        "100": 100.0,
        "5 * 2 + 3 * 4": 22.0, # 10 + 12 = 22
        "1 + 2 - 3 + 4": 4.0,   # (1+2)-3+4 = 3-3+4 = 0+4 = 4
        "10 / (2 + 3) * 2": 4.0, # 10 / 5 * 2 = 2 * 2 = 4
        "7": 7.0,
        "5 - 2": 3.0,
        "2 * 3": 6.0
    }

    for expr_str, expected_result in evaluation_tests.items():
        print(f"Expression: '{expr_str}'")
        try:
            result = evaluate_expression(expr_str)
            print(f"  Expected: {expected_result}, Got: {result}, Match: {result == expected_result}")
            if result != expected_result:
                 print(f"  MISMATCH! RPN was: {parse_expression(expr_str)}")
        except ValueError as e:
            print(f"  Error during evaluation: {e}")

    print("\nTesting evaluate_expression (with expected errors):")
    error_evaluation_tests = [
        "1 + a",                  # Invalid character
        "10 / 0",                 # Division by zero
        "(1 + 2",                 # Mismatched parenthesis (from parse_expression)
        "1 +",                    # Insufficient operands (will fail in parse or RPN eval)
                                  # parse_expression for "1 +" -> [1, '+']
                                  # evaluate_expression will find insufficient operands for '+'
        "1 2 +",                  # Valid RPN if input directly, but not from parse_expression for "1 2 +"
                                  # If expression is "1 2", parse_expression might give [1, 2]
                                  # which evaluate_expression will flag as invalid final stack.
        "eval('1+1')",            # Invalid character
        "1 / (2 - 2)",            # Division by zero
        "((1+2)*3",               # Mismatched parenthesis
        "1 * (2 + )"              # parse_expression for "1 * (2 + )" -> [1, 2, '+', '*']
                                  # This is (1 * (2+)). Evaluator will try 2 + (nothing) -> error
                                  # Or if it means (1*(2+X)), then it's incomplete.
                                  # The RPN [1, 2, '+', '*'] will evaluate as:
                                  # stack: [1]
                                  # stack: [1, 2]
                                  # token: '+', pop 2, pop 1. Need 2 operands. Error.
                                  # Actually, for "1 * (2+)", RPN is [1, 2, '+', '*']
                                  # Stack: [1]
                                  # Stack: [1, 2.0]
                                  # Token: '+', op2=2.0, op1=1.0. result = 3.0. Stack: [3.0]
                                  # Token: '*', op2=3.0. Need op1. Error: Insufficient operands.
                                  # Let's re-check parse_expression for "1 * (2 + )"
                                  # tokens: ['1', '*', '(', '2', '+', ')']
                                  # '1': output_queue: [1]
                                  # '*': op_stack: ['*']
                                  # '(': op_stack: ['*', '(']
                                  # '2': output_queue: [1, 2]
                                  # '+': op_stack: ['*', '(', '+'] (since prec('+') < prec('(') is false, prec('+') > prec('*') is false)
                                  #      Actually, prec('+')=1, prec of stack top '(' is not compared.
                                  #      prec('+')=1, stack top is '('. Push '+'. op_stack: ['*', '(', '+']
                                  # ')': pop '+', output_queue: [1, 2, '+']
                                  #      pop '(', op_stack: ['*']
                                  # end: pop '*', output_queue: [1, 2, '+', '*'] - This is correct for (1+2)* if it were infix.
                                  #      No, this is RPN for 1 2 + * which is (1+2)*.
                                  #      The expression was 1 * (2+).
                                  #      Let's trace parse_expression for "1 * (2 + )" again.
                                  #      processed_expression = "1*(2+)"
                                  #      tokens = ['1', '*', '(', '2', '+', ')']
                                  #      token '1': output_queue = [1]
                                  #      token '*': operator_stack = ['*']
                                  #      token '(': operator_stack = ['*', '(']
                                  #      token '2': output_queue = [1, 2]
                                  #      token '+': operator_stack = ['*', '(', '+'] (precedence of '+' (1) vs stack top '(' -> push)
                                  #      token ')':
                                  #          while operator_stack and operator_stack[-1] != '(':
                                  #              pop '+', output_queue = [1, 2, '+']
                                  #              operator_stack = ['*', '(']
                                  #          pop '(', operator_stack = ['*']
                                  #      end of tokens.
                                  #      while operator_stack:
                                  #          pop '*', output_queue = [1, 2, '+', '*']
                                  #      RPN: [1, 2, '+', '*']
                                  #      Evaluation:
                                  #      1 -> stack [1.0]
                                  #      2 -> stack [1.0, 2.0]
                                  #      + -> op2=2.0, op1=1.0, res=3.0. stack [3.0]
                                  #      * -> op2=3.0. Need op1. Error: "Insufficient operands for operator '*'". This is correct.
        "1 2"                     # Valid chars, parse_expression gives [1, 2]. evaluate_expression will error (stack > 1).
    ]

    for expr_str in error_evaluation_tests:
        print(f"Expression: '{expr_str}'")
        try:
            result = evaluate_expression(expr_str)
            print(f"  Unexpected success. Got: {result}")
        except ValueError as e:
            print(f"  Expected error: {e}")
