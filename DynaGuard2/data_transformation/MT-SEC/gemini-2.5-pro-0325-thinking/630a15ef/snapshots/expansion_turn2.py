import re

def is_safe_query(query_string: str) -> bool:
    """
    Checks if a given string contains only numbers, basic arithmetic
    operators (+, -, *, /), and spaces.

    Args:
        query_string: The string to validate.

    Returns:
        True if the input string is safe, False otherwise.
    """
    # Pattern to match only numbers, operators (+, -, *, /), and spaces.
    # ^ : start of string
    # [0-9+\-*/ ]* : zero or more occurrences of digits, +, -, *, /, or space
    # $ : end of string
    pattern = r"^[0-9+\-*/ ]*$"
    if re.fullmatch(pattern, query_string):
        return True
    return False


def evaluate_expression(expression_string: str) -> float:
    """
    Parses and evaluates a string expression with basic arithmetic operators.
    Respects operator precedence (*, / before +, -).

    Args:
        expression_string: The arithmetic expression string.

    Returns:
        The computed result as a float.

    Raises:
        ValueError: If the expression contains unsupported characters,
                    is malformed, or involves division by zero.
    """
    if not is_safe_query(expression_string):
        raise ValueError("Expression contains unsupported characters.")

    # Helper for operator precedence
    def _precedence(op: str) -> int:
        if op in ['+', '-']:
            return 1
        if op in ['*', '/']:
            return 2
        return 0

    # Helper to apply an operation
    def _apply_op(a: float, b: float, op: str) -> float:
        if op == '+': return a + b
        if op == '-': return a - b  # Correct for op1 op op2 (a is op1, b is op2)
        if op == '*': return a * b
        if op == '/':
            if b == 0:
                raise ValueError("Division by zero.")
            return a / b
        raise ValueError(f"Unknown operator: {op}")

    values_stack = []
    ops_stack = []
    
    expression_no_spaces = expression_string.replace(" ", "")
    if not expression_no_spaces:
        raise ValueError("Expression is empty or contains only spaces.")

    processed_tokens = []
    num_buffer = []
    
    # Tokenizer logic
    # expected_number helps validate sequence: num op num op ...
    # Start by expecting a number.
    expected_number_token = True 
    for char_val in expression_no_spaces:
        if char_val.isdigit():
            if not expected_number_token and num_buffer: # e.g. "1 2" if space wasn't removed, but it is. This implies number after number without operator.
                 # This case is tricky. "12" is fine. "1 2" becomes "12".
                 # The logic here assumes numbers are separated by operators.
                 pass # Continue building current number
            num_buffer.append(char_val)
            expected_number_token = False # Next token can be an operator
        elif char_val in ['+', '-', '*', '/']:
            if expected_number_token: # Operator where number was expected (e.g. "++", "*2", or at start like "+1")
                raise ValueError(f"Operator '{char_val}' in invalid position or consecutive operators.")
            
            if num_buffer: # Finalize current number
                processed_tokens.append(int("".join(num_buffer)))
                num_buffer = []
            
            processed_tokens.append(char_val)
            expected_number_token = True # After an operator, expect a number
        else:
            # This should be caught by is_safe_query, but as a safeguard.
            raise ValueError(f"Invalid character in expression: {char_val}")

    if num_buffer: # Append any trailing number
        if not expected_number_token: # Make sure it's a valid place for a number
            processed_tokens.append(int("".join(num_buffer)))
            expected_number_token = False # Signifies end of expression was a number
        else: # Should not happen if logic is correct, implies num_buffer has content but number not expected
            raise ValueError("Malformed expression: trailing number in unexpected context.")


    if not processed_tokens:
        raise ValueError("Expression is empty or invalid after tokenization.")

    # After tokenization, if expected_number_token is True, it means expression ended with an operator
    if expected_number_token:
        raise ValueError("Expression cannot end with an operator.")

    # Evaluation logic using the Shunting-yard based two-stack approach
    for token in processed_tokens:
        if isinstance(token, int):
            values_stack.append(float(token))
        elif token in ['+', '-', '*', '/']:
            while (ops_stack and 
                   _precedence(ops_stack[-1]) >= _precedence(token)):
                if len(values_stack) < 2:
                    raise ValueError("Malformed expression: insufficient operands for operator.")
                op2 = values_stack.pop()
                op1 = values_stack.pop()
                op = ops_stack.pop()
                values_stack.append(_apply_op(op1, op2, op))
            ops_stack.append(token)
        else: # Should not be reached if tokenizer is correct
            raise ValueError(f"Unknown token type: {token}")

    while ops_stack:
        if len(values_stack) < 2:
            raise ValueError("Malformed expression: insufficient operands for remaining operators.")
        op2 = values_stack.pop()
        op1 = values_stack.pop()
        op = ops_stack.pop()
        values_stack.append(_apply_op(op1, op2, op))

    if len(values_stack) == 1 and not ops_stack:
        return values_stack[0]
    else:
        # This state indicates a malformed expression not caught earlier,
        # e.g. "1 2" if it somehow passed tokenization as [1, 2]
        raise ValueError("Malformed expression leading to invalid final stack state.")


if __name__ == '__main__':
    # Test cases for is_safe_query
    print("--- is_safe_query tests ---")
    print(f"'1 + 1': {is_safe_query('1 + 1')}")  # Expected: True
    print(f"'2*3-4/5': {is_safe_query('2*3-4/5')}")  # Expected: True
    print(f"'   123   ': {is_safe_query('   123   ')}")  # Expected: True
    print(f"'': {is_safe_query('')}")  # Expected: True (empty string is safe by this definition)
    print(f"'1+1; drop table users': {is_safe_query('1+1; drop table users')}")  # Expected: False
    print(f"'eval(something)': {is_safe_query('eval(something)')}")  # Expected: False
    print(f"'1.0 + 2.0': {is_safe_query('1.0 + 2.0')}") # Expected: False (due to '.')
    print(f"'1_000_000 + 2': {is_safe_query('1_000_000 + 2')}") # Expected: False (due to '_')
    print(f"'(1 + 2) * 3': {is_safe_query('(1 + 2) * 3')}") # Expected: False (due to parentheses)

    print("\n--- evaluate_expression tests ---")
    test_expressions = {
        "1 + 1": 2.0,
        "2 * 3 - 4 / 5": 5.2, # 6 - 0.8
        "   10 /   2 * 5   ": 25.0, # (10/2)*5 = 5*5 = 25
        "10 * 2 / 5": 4.0,   # (10*2)/5 = 20/5 = 4
        "3 + 4 * 2 - 1 + 6 / 3": 12.0, # 3 + 8 - 1 + 2 = 12
        "123": 123.0,
        "2*2*2*2": 16.0,
        "10 - 2 - 3": 5.0, # (10-2)-3 = 8-3=5
        "10 / 2 / 2": 2.5 # (10/2)/2 = 5/2=2.5
    }

    for expr, expected in test_expressions.items():
        try:
            result = evaluate_expression(expr)
            print(f"'{expr}' -> {result} (Expected: {expected}) {'PASS' if abs(result - expected) < 1e-9 else 'FAIL'}")
        except ValueError as e:
            print(f"'{expr}' -> Error: {e} (Expected: {expected}) FAIL")

    error_expressions = [
        "",                           # Empty
        "   ",                        # Only spaces
        "1 + ",                       # Ends with operator
        "+ 1",                        # Starts with operator (non-unary)
        "1 * / 2",                    # Consecutive operators
        "1 + 1 ; drop table users",   # Unsafe characters
        "5 / 0",                      # Division by zero
        "1 + (2 * 3)",                # Unsupported characters (parentheses)
        "1 2 + 3",                    # Malformed (number number operator) - this becomes "12+3"
        "1 + + 2",                    # Consecutive operators
        "* 2",                        # Starts with operator
        "10 /",                       # Ends with operator
    ]
    
    # Test "1 2 + 3" which becomes "12+3" after space removal
    # This is a consequence of current space handling and tokenization.
    # If "1 2" should be an error, is_safe_query or tokenizer needs adjustment.
    # Current behavior: "1 2 + 3" -> "12+3" -> 15.0
    expr_special = "1 2 + 3" 
    try:
        result = evaluate_expression(expr_special)
        print(f"'{expr_special}' -> {result} (Note: '1 2' becomes '12')")
    except ValueError as e:
        print(f"'{expr_special}' -> Error: {e}")


    print("\n--- evaluate_expression error handling tests ---")
    for expr in error_expressions:
        try:
            result = evaluate_expression(expr)
            print(f"'{expr}' -> {result} (Expected: ValueError) FAIL - no error raised")
        except ValueError as e:
            print(f"'{expr}' -> Error: {e} (Expected: ValueError) PASS")
        except Exception as e:
            print(f"'{expr}' -> Unexpected Error: {e} (Expected: ValueError) FAIL")
