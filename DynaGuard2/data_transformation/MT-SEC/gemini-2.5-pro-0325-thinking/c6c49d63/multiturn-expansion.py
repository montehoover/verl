import re
import ast

def is_valid_expression(expression_string: str) -> bool:
    """
    Checks if a given string contains only numbers, basic arithmetic
    operators (+, -, *, /), parentheses, and spaces.

    Args:
        expression_string: The string to validate.

    Returns:
        True if the input string is a valid expression, False otherwise.
    """
    # Regex to allow numbers, operators (+, -, *, /), parentheses, and spaces.
    # ^ : asserts position at start of the string.
    # [\d\s\+\-\*\/\(\)]* : matches any character in the set (digits, spaces, +, -, *, /, (, )) zero or more times.
    # $ : asserts position at the end of the string.
    # The operators * and + need to be escaped with \ as they have special meaning in regex.
    # Parentheses ( and ) also need to be escaped.
    # - (hyphen) when not at the start or end of a character set, or not part of a range,
    #   can be used literally, but escaping it as \- is safer for clarity.
    # / (slash) does not need escaping in Python's re module unless it's used as a delimiter (not the case here).
    #
    # More precise pattern: allow numbers (integers and floats), operators, parentheses, and spaces.
    # This pattern ensures that only these characters are present.
    # It doesn't validate the *grammatical* correctness of the expression (e.g., "2 + * 3" would pass this character check).
    # For stricter validation of allowed characters:
    # \d for digits
    # \s for whitespace
    # \. for literal dot (for floating point numbers)
    # \+ for plus
    # \- for minus
    # \* for asterisk
    # \/ for slash
    # \( for open parenthesis
    # \) for close parenthesis
    allowed_chars_pattern = r"^[0-9\s\+\-\*\/\(\)\.]*$"

    if re.fullmatch(allowed_chars_pattern, expression_string):
        return True
    return False


def apply_operator(op1: float, op2: float, operator: str) -> float:
    """
    Applies the given arithmetic operator to the two operands.

    Args:
        op1: The first operand.
        op2: The second operand.
        operator: The operator string ('+', '-', '*', '/').

    Returns:
        The result of the operation.

    Raises:
        ValueError: If the operator is unsupported.
        ZeroDivisionError: If division by zero is attempted.
    """
    if operator == '+':
        return op1 + op2
    elif operator == '-':
        return op1 - op2
    elif operator == '*':
        return op1 * op2
    elif operator == '/':
        if op2 == 0:
            raise ZeroDivisionError("Division by zero")
        return op1 / op2
    else:
        raise ValueError(f"Unsupported operator: {operator}")


def evaluate_expression(tokens: list) -> float:
    """
    Evaluates a list of tokens (numbers and operators), respecting operator precedence.
    Handles basic arithmetic operators (+, -, *, /) and parentheses.

    Args:
        tokens: A list of numbers (int or float) and strings (operators or parentheses).
                Example: [3, '+', 4, '*', '(', 2, '-', 1, ')']

    Returns:
        The result of the evaluated expression.

    Raises:
        ValueError: If the expression is malformed, contains unsupported operators,
                    or has mismatched parentheses.
        ZeroDivisionError: If division by zero occurs.
    """
    values_stack = []
    ops_stack = []

    def _precedence(op: str) -> int:
        if op == '+' or op == '-':
            return 1
        if op == '*' or op == '/':
            return 2
        return 0  # For parentheses

    def _process_top_operator():
        try:
            op = ops_stack.pop()
            val2 = values_stack.pop()
            val1 = values_stack.pop()
            values_stack.append(apply_operator(val1, val2, op))
        except IndexError:
            raise ValueError("Invalid expression structure or insufficient operands for an operator")


    for token in tokens:
        if isinstance(token, (int, float)):
            values_stack.append(float(token))
        elif token == '(':
            ops_stack.append(token)
        elif token == ')':
            while ops_stack and ops_stack[-1] != '(':
                _process_top_operator()
            if not ops_stack or ops_stack[-1] != '(':
                raise ValueError("Mismatched parentheses: missing '('")
            ops_stack.pop()  # Pop '('
        elif token in ['+', '-', '*', '/']:
            while (ops_stack and ops_stack[-1] != '(' and
                   _precedence(ops_stack[-1]) >= _precedence(token)):
                _process_top_operator()
            ops_stack.append(token)
        else:
            raise ValueError(f"Invalid token in expression: {token}")

    while ops_stack:
        if ops_stack[-1] == '(':
            raise ValueError("Mismatched parentheses: missing ')'")
        _process_top_operator()

    if len(values_stack) == 1:
        return values_stack[0]
    elif not values_stack and not tokens: # Empty initial token list and nothing processed
        raise ValueError("Cannot evaluate an empty expression")
    else:
        # This state can be reached for malformed expressions like "1 2" (no operator)
        # or if the initial token list was empty and somehow bypassed earlier checks.
        raise ValueError("Invalid expression structure or empty expression")


def _tokenize_expression(expression_string: str) -> list:
    """
    Tokenizes an arithmetic expression string into a list of numbers (float) and operators/parentheses (str).
    Handles integers, floats, and negative numbers.
    Removes whitespace before tokenizing.
    """
    processed_expr = expression_string.replace(" ", "")
    if not processed_expr:
        return []

    # Regex to find:
    # 1. Floats (e.g., -3.14, 3.14, .5, -.5)
    # 2. Integers (e.g., -5, 5)
    # 3. Operators and parentheses (+, -, *, /, (, ))
    # Order matters: floats before integers to correctly parse numbers like "3.14".
    # Pattern for numbers allows for optional leading minus, and optional integer part for floats (e.g. .5 or -.5)
    # or optional fractional part for integers that become floats (e.g. 5. -> 5.0)
    # A more robust number regex: -?(?:\d+\.\d*|\.\d+|\d+)
    #   -? : optional minus
    #   (?: ... ) : non-capturing group
    #   \d+\.\d* : one or more digits, a dot, zero or more digits (e.g., "3.", "3.14")
    #   |\.\d+   : OR a dot, one or more digits (e.g., ".5")
    #   |\d+     : OR one or more digits (e.g., "5")
    # This combined with operators:
    token_pattern_str = r"(-?(?:\d+\.\d*|\.\d+|\d+)|[+\-*/()])"
    # Example: "-5.5+2" -> ["-5.5", "+", "2"]
    # Example: ".5*-2" -> [".5", "*", "-2"]

    raw_tokens = re.findall(token_pattern_str, processed_expr)

    # Validate that the entire string was tokenized.
    # This catches cases like "1.2.3" which is valid by is_valid_expression (char-wise)
    # but is not a valid sequence of numbers/operators for this tokenizer.
    # Or "1a" where 'a' is not tokenized.
    # is_valid_expression should ideally catch "1a", but this is a safeguard.
    if "".join(raw_tokens) != processed_expr:
        # Attempt to find the first untokenized part for a better error message
        untokenized_check = ""
        last_match_end = 0
        for match in re.finditer(token_pattern_str, processed_expr):
            if match.start() != last_match_end:
                untokenized_part = processed_expr[last_match_end:match.start()]
                raise ValueError(f"Invalid part in expression starting with: '{untokenized_part}'")
            untokenized_check += match.group(0)
            last_match_end = match.end()
        if last_match_end != len(processed_expr):
             raise ValueError(f"Invalid part at end of expression: '{processed_expr[last_match_end:]}'")
        # Fallback generic error if specific part not found (should not happen if logic above is correct)
        raise ValueError(f"Invalid characters or structure in expression: {expression_string}")


    tokens = []
    for token_str in raw_tokens:
        try:
            tokens.append(float(token_str))
        except ValueError:
            tokens.append(token_str)  # Operator or parenthesis
    return tokens


def compute_expression(input_expr: str) -> float:
    """
    Computes the result of an arithmetic expression string.

    Args:
        input_expr: The arithmetic expression string.

    Returns:
        The computed result as a float.

    Raises:
        ValueError: If the input expression is invalid, contains unsupported characters,
                    is malformed, or results in an error during evaluation (e.g. mismatched parentheses).
        ZeroDivisionError: If division by zero occurs.
    """
    if not is_valid_expression(input_expr):
        # This check is somewhat redundant if _tokenize_expression's validation is robust,
        # but good for early exit on clearly invalid characters.
        raise ValueError(f"Expression contains invalid characters: {input_expr}")

    tokens = _tokenize_expression(input_expr)

    # evaluate_expression will raise ValueError for an empty token list
    # (e.g. if input_expr was "" or "   ")
    return evaluate_expression(tokens)


if __name__ == '__main__':
    # Test cases for is_valid_expression
    print("--- is_valid_expression tests ---")
    print(f"'1 + 2 * (3 / 4) - 5': {is_valid_expression('1 + 2 * (3 / 4) - 5')}")  # Expected: True
    print(f"'10.5 * 2': {is_valid_expression('10.5 * 2')}") # Expected: True
    print(f"'   ( ( 1+2 ) * 3 )   ': {is_valid_expression('   ( ( 1+2 ) * 3 )   ')}") # Expected: True
    print(f"'-5 + (3*9)': {is_valid_expression('-5 + (3*9)')}") # Expected: True
    print(f"'1 + 2a': {is_valid_expression('1 + 2a')}")  # Expected: False
    print(f"'eval(something)': {is_valid_expression('eval(something)')}")  # Expected: False
    print(f"'1 & 2': {is_valid_expression('1 & 2')}")  # Expected: False
    print(f"'1; drop table users': {is_valid_expression('1; drop table users')}") # Expected: False
    print(f"Empty string '': {is_valid_expression('')}") # Expected: True
    print(f"Only spaces '   ': {is_valid_expression('   ')}") # Expected: True
    print(f"Invalid char '^': {is_valid_expression('1 ^ 2')}") # Expected: False

    # Test cases for apply_operator
    print("\n--- apply_operator tests ---")
    print(f"3 + 5 = {apply_operator(3, 5, '+')}")  # Expected: 8.0
    print(f"10 - 4 = {apply_operator(10, 4, '-')}")  # Expected: 6.0
    print(f"6 * 7 = {apply_operator(6, 7, '*')}")  # Expected: 42.0
    print(f"20 / 5 = {apply_operator(20, 5, '/')}")  # Expected: 4.0
    try:
        apply_operator(1, 0, '/')
    except ZeroDivisionError as e:
        print(f"1 / 0: {e}")  # Expected: Division by zero
    try:
        apply_operator(1, 2, '%')
    except ValueError as e:
        print(f"1 % 2: {e}")  # Expected: Unsupported operator: %

    # Test cases for evaluate_expression
    print("\n--- evaluate_expression tests ---")
    print(f"eval([1, '+', 2, '*', 3]): {evaluate_expression([1, '+', 2, '*', 3])}")  # Expected: 7.0
    print(f"eval([10, '/', 2, '+', 3]): {evaluate_expression([10, '/', 2, '+', 3])}")  # Expected: 8.0
    print(f"eval([10, '+', 2, '/', 2]): {evaluate_expression([10, '+', 2, '/', 2])}")  # Expected: 11.0
    print(f"eval(['(', 1, '+', 2, ')', '*', 3]): {evaluate_expression(['(', 1, '+', 2, ')', '*', 3])}")  # Expected: 9.0
    print(f"eval([5]): {evaluate_expression([5])}")  # Expected: 5.0
    print(f"eval([3, '*', '(', 4, '+', 2, ')']): {evaluate_expression([3, '*', '(', 4, '+', 2, ')'])}")  # Expected: 18.0
    
    # More complex expression: (2 + 3) * 4 - 12 / 3 + (5 - 1) = 5 * 4 - 4 + 4 = 20 - 4 + 4 = 20
    expr_tokens = ['(', 2, '+', 3, ')', '*', 4, '-', 12, '/', 3, '+', '(', 5, '-', 1, ')']
    print(f"eval({expr_tokens}): {evaluate_expression(expr_tokens)}")  # Expected: 20.0

    # Test for negative numbers (assuming tokenizer handles unary minus to make numbers negative)
    # For example, "-5 + 2" would be tokenized as [-5, '+', 2]
    print(f"eval([-5, '+', 2]): {evaluate_expression([-5, '+', 2])}") # Expected: -3.0
    print(f"eval([2, '*', -3]): {evaluate_expression([2, '*', -3])}") # Expected: -6.0


    print("\n--- evaluate_expression error handling tests ---")
    try:
        evaluate_expression([])
    except ValueError as e:
        print(f"eval([]): {e}") # Expected: Cannot evaluate an empty expression

    try:
        evaluate_expression([1, '+']) 
    except ValueError as e: 
        print(f"eval([1, '+']): Error - {e}")

    try:
        evaluate_expression(['(', 1, '+', 2]) 
    except ValueError as e:
        print(f"eval(['(', 1, '+', 2]): {e}") # Expected: Mismatched parentheses: missing ')'

    try:
        evaluate_expression([1, '+', ')'])
    except ValueError as e:
        print(f"eval([1, '+', ')']): {e}") # Expected: Mismatched parentheses: missing '('
        
    try:
        evaluate_expression([1, '+', 'a'])
    except ValueError as e:
        print(f"eval([1, '+', 'a']): {e}") # Expected: Invalid token in expression: a

    try:
        evaluate_expression([1, '/', 0])
    except ZeroDivisionError as e:
        print(f"eval([1, '/', 0]): {e}") # Expected: Division by zero
        
    try:
        evaluate_expression([1, 2, '+']) # Should be an error due to structure
    except ValueError as e:
        print(f"eval([1, 2, '+']): {e}")
        
    try:
        evaluate_expression(['+']) # Should be an error
    except ValueError as e:
        print(f"eval(['+']): {e}")

    # Test cases for compute_expression
    print("\n--- compute_expression tests ---")
    print(f"compute_expression('1 + 2 * 3') = {compute_expression('1 + 2 * 3')}")  # Expected: 7.0
    print(f"compute_expression('(1 + 2) * 3') = {compute_expression('(1 + 2) * 3')}")  # Expected: 9.0
    print(f"compute_expression('10 / 2 + 3') = {compute_expression('10 / 2 + 3')}")  # Expected: 8.0
    print(f"compute_expression('10 + 2 / 2') = {compute_expression('10 + 2 / 2')}")  # Expected: 11.0
    print(f"compute_expression('3 * (4 + 2)') = {compute_expression('3 * (4 + 2)')}") # Expected: 18.0
    print(f"compute_expression('-5 + 2') = {compute_expression('-5 + 2')}") # Expected: -3.0
    print(f"compute_expression('5 + -2') = {compute_expression('5 + -2')}") # Expected: 3.0
    print(f"compute_expression('5 * -2') = {compute_expression('5 * -2')}") # Expected: -10.0
    print(f"compute_expression('(2 + 3) * 4 - 12 / 3 + (5 - 1)') = {compute_expression('(2 + 3) * 4 - 12 / 3 + (5 - 1)')}") # Expected: 20.0
    print(f"compute_expression('10.5 * 2') = {compute_expression('10.5 * 2')}") # Expected: 21.0
    print(f"compute_expression('.5 * 4') = {compute_expression('.5 * 4')}") # Expected: 2.0
    print(f"compute_expression('-.5 * 4') = {compute_expression('-.5 * 4')}") # Expected: -2.0
    print(f"compute_expression('5.') = {compute_expression('5.')}") # Expected: 5.0

    print("\n--- compute_expression error handling tests ---")
    try:
        compute_expression("1 + 2a")
    except ValueError as e:
        print(f"compute_expression('1 + 2a'): {e}") # Expected: Expression contains invalid characters... or Invalid part...

    try:
        compute_expression("1 +")
    except ValueError as e:
        print(f"compute_expression('1 +'): {e}") # Expected: Invalid expression structure...

    try:
        compute_expression("(1 + 2")
    except ValueError as e:
        print(f"compute_expression('(1 + 2'): {e}") # Expected: Mismatched parentheses...

    try:
        compute_expression("1 / 0")
    except ZeroDivisionError as e:
        print(f"compute_expression('1 / 0'): {e}") # Expected: Division by zero

    try:
        compute_expression("") # Valid chars, but empty expression
    except ValueError as e:
        print(f"compute_expression(''): {e}") # Expected: Cannot evaluate an empty expression

    try:
        compute_expression("   ") # Valid chars, but empty expression
    except ValueError as e:
        print(f"compute_expression('   '): {e}") # Expected: Cannot evaluate an empty expression
        
    try:
        compute_expression("1 . 2 + 3") # is_valid_expression allows '.', but tokenizer should fail
    except ValueError as e:
        print(f"compute_expression('1 . 2 + 3'): {e}")

    try:
        compute_expression("1.2.3") # is_valid_expression allows '.', but tokenizer should fail
    except ValueError as e:
        print(f"compute_expression('1.2.3'): {e}")
