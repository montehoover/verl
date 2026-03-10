import re

def sanitize_input(expression: str) -> bool:
    """
    Checks if the input string contains only numbers, arithmetic operators,
    parentheses, decimal points, and spaces.

    Args:
        expression: The string to sanitize.

    Returns:
        True if the input string is safe.

    Raises:
        ValueError: If the input string contains invalid characters.
    """
    # Regex to allow numbers (including decimals), operators (+, -, *, /),
    # parentheses, and spaces.
    # ^ : start of string
    # [0-9] : digits
    # \. : literal dot (for decimal numbers)
    # \+ : literal +
    # \- : literal -
    # \* : literal *
    # \/ : literal /
    # \( : literal (
    # \) : literal )
    # \s : whitespace characters
    # * : zero or more occurrences of the preceding characters
    # $ : end of string
    pattern = r"^[0-9\.\+\-\*\/\(\)\s]*$"

    if re.fullmatch(pattern, expression):
        return True
    else:
        # Find the invalid characters to include in the error message for clarity.
        invalid_chars = "".join(sorted(list(set(char for char in expression if not re.match(r"[0-9\.\+\-\*\/\(\)\s]", char)))))
        raise ValueError(f"Input contains invalid characters: '{invalid_chars}'. Only numbers, operators (+, -, *, /), parentheses, decimal points, and spaces are allowed.")

def parse_expression(expression: str) -> list:
    """
    Parses a sanitized arithmetic expression string into a list of tokens
    (numbers, operators, parentheses).

    Args:
        expression: The sanitized arithmetic expression string.

    Returns:
        A list of tokens (strings and floats).
        Operators and parentheses are strings. Numbers are floats.
    """
    # Remove all whitespace to simplify tokenization
    expression = expression.replace(" ", "")

    # Regex to find numbers (including decimals) or operators/parentheses
    # Order matters: match multi-character tokens first (e.g., numbers)
    # then single-character tokens (operators, parentheses).
    # (\d+\.\d+) : matches floating point numbers like 123.45
    # (\d+) : matches integers like 123
    # ([+\-*/()]) : matches operators +, -, *, /, or parentheses ( )
    token_pattern = re.compile(r"(\d+\.\d+|\d+|[+\-*/()])")
    tokens = token_pattern.findall(expression)

    # Convert numeric tokens to float, leave operators/parentheses as strings
    processed_tokens = []
    for token in tokens:
        if token.isdigit() or ('.' in token and token.replace('.', '', 1).isdigit()):
            processed_tokens.append(float(token))
        elif token in ['+', '-', '*', '/', '(', ')']:
            processed_tokens.append(token)
        else:
            # This case should ideally not be reached if sanitize_input was effective
            # and the regex is correct.
            raise ValueError(f"Unexpected token during parsing: {token}")
    return processed_tokens

# --- Evaluation Logic ---

_EVAL_PRECEDENCE = {'+': 1, '-': 1, '*': 2, '/': 2}

def _apply_operator_for_eval(operators_stack: list, operands_stack: list):
    """
    Applies an operator from the operators_stack to operands from the operands_stack.
    Modifies operands_stack with the result.
    Raises ValueError for insufficient operands or unknown operator.
    Raises ZeroDivisionError for division by zero.
    """
    if not operators_stack:
        # This should ideally not be hit if the calling logic is correct (e.g. _evaluate_parsed_tokens)
        raise ValueError("Invalid expression: Attempted to apply operator from an empty stack.")
    
    op = operators_stack.pop()

    # All current operators are binary. Unary minus is handled by pushing a '0' operand earlier.
    if len(operands_stack) < 2:
        raise ValueError(f"Invalid expression: Not enough operands for operator '{op}'.")
    
    right_operand = operands_stack.pop()
    left_operand = operands_stack.pop()

    if op == '+':
        operands_stack.append(left_operand + right_operand)
    elif op == '-':
        operands_stack.append(left_operand - right_operand)
    elif op == '*':
        operands_stack.append(left_operand * right_operand)
    elif op == '/':
        if right_operand == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        operands_stack.append(left_operand / right_operand)
    else:
        # Should not happen if operators are restricted to _EVAL_PRECEDENCE keys
        raise ValueError(f"Unknown operator encountered during evaluation: {op}")

def _evaluate_parsed_tokens(tokens: list) -> float:
    """
    Evaluates a list of tokens (numbers and operators) to a single float result.
    Uses Shunting-yard-like logic for handling precedence and parentheses.
    Args:
        tokens: A list of floats and strings (operators/parentheses).
    Returns:
        The numerical result of the expression.
    Raises:
        ValueError: For malformed expressions (e.g., mismatched parentheses, insufficient operands).
        ZeroDivisionError: For division by zero.
    """
    operands_stack = []
    operators_stack = []
    # True if the context allows for a unary operator (e.g. at start, after '(', after another operator)
    allow_unary_operator = True 

    for token in tokens:
        if isinstance(token, float):
            operands_stack.append(token)
            allow_unary_operator = False
        elif token == '(':
            operators_stack.append(token)
            allow_unary_operator = True
        elif token == ')':
            while operators_stack and operators_stack[-1] != '(':
                _apply_operator_for_eval(operators_stack, operands_stack)
            if not operators_stack or operators_stack[-1] != '(':
                raise ValueError("Mismatched parentheses: Missing '('.")
            operators_stack.pop() # Pop '('
            allow_unary_operator = False 
        elif token in _EVAL_PRECEDENCE: # Operator
            if token == '-' and allow_unary_operator:
                operands_stack.append(0.0) # Handle unary minus by making it 0 - number
            
            while (operators_stack and operators_stack[-1] != '(' and
                   _EVAL_PRECEDENCE.get(operators_stack[-1], 0) >= _EVAL_PRECEDENCE.get(token, 0)):
                _apply_operator_for_eval(operators_stack, operands_stack)
            operators_stack.append(token)
            allow_unary_operator = True
        else:
            raise ValueError(f"Unexpected token during evaluation: {token}")

    while operators_stack:
        if operators_stack[-1] == '(':
            raise ValueError("Mismatched parentheses: Missing ')'.")
        _apply_operator_for_eval(operators_stack, operands_stack)

    if len(operands_stack) == 1:
        return operands_stack[0]
    elif not operands_stack and not tokens: # Handles empty expression string
        raise ValueError("Invalid expression: Expression is empty.")
    else:
        # Catches cases like "1 2" or "()" or if expression is just operators.
        raise ValueError("Invalid expression format or insufficient operands for a complete calculation.")

def safe_eval_expression(expression_string: str) -> float:
    """
    Sanitizes, parses, and evaluates a mathematical expression string.
    Handles basic arithmetic (+, -, *, /) and parentheses.
    Args:
        expression_string: The mathematical expression string.
    Returns:
        The result of the mathematical evaluation as a float.
    Raises:
        ValueError: If the input string is deemed unsafe, malformed,
                    or if there's an issue during parsing or evaluation
                    (e.g., mismatched parentheses, insufficient operands).
        ZeroDivisionError: If division by zero is attempted.
    """
    # 1. Sanitize input
    # sanitize_input will raise ValueError if invalid characters are found.
    # No need for a try-except block here if we want sanitize_input's error to propagate directly.
    # However, to provide a more specific error context from safe_eval_expression:
    try:
        sanitize_input(expression_string)
    except ValueError as e:
        # Augment the error message or re-raise if preferred.
        raise ValueError(f"Input sanitization failed: {e}")

    # 2. Parse expression into tokens
    # parse_expression can also raise ValueError for unexpected tokens (though sanitize should prevent this)
    try:
        tokens = parse_expression(expression_string)
    except ValueError as e:
        raise ValueError(f"Expression parsing failed: {e}")

    # 3. Evaluate the token list
    # _evaluate_parsed_tokens will raise ValueError or ZeroDivisionError for evaluation issues.
    try:
        result = _evaluate_parsed_tokens(tokens)
        return result
    except (ValueError, ZeroDivisionError) as e:
        # Re-raise evaluation errors with potentially more context if needed, or just let them propagate.
        # For example, could add the original expression string to the error.
        raise type(e)(f"Evaluation error for '{expression_string}': {e}")


if __name__ == '__main__':
    # Example Usage for sanitize_input
    print("--- Testing sanitize_input ---")
    test_expressions_sanitize = [
        "1 + 2",
        " (3.14 * 2) - 7 / (4 + 1) ",
        "12345",
        "-5.0",
        "100 / 0.5",
        "", # Empty string
        "1 + 2$", # Invalid character $
        "abc",    # Invalid characters a, b, c
        "1 + (2 * 3!", # Invalid character !
        "eval('os.system(\"reboot\")')", # Malicious attempt
    ]

    for expr in test_expressions_sanitize:
        try:
            is_safe = sanitize_input(expr)
            print(f"Sanitize Expression: '{expr}' -> Safe: {is_safe}")
        except ValueError as e:
            print(f"Sanitize Expression: '{expr}' -> Error: {e}")

    print("\n--- Testing sanitize_input specific cases ---")
    try:
        sanitize_input("2+2") # Valid
        print("'2+2' is valid.")
    except ValueError:
        print("'2+2' raised ValueError unexpectedly.")

    try:
        sanitize_input("2+2a") # Invalid
        print("'2+2a' did not raise ValueError as expected.")
    except ValueError as e:
        print(f"'2+2a' raised ValueError as expected: {e}")

    # Example Usage for parse_expression
    print("\n--- Testing parse_expression ---")
    test_expressions_parse = [
        ("1 + 2", [1.0, '+', 2.0]),
        ("(3.14 * 2) - 7 / (4 + 1)", ['(', 3.14, '*', 2.0, ')', '-', 7.0, '/', '(', 4.0, '+', 1.0, ')']),
        ("12345", [12345.0]),
        ("-5.0", ['-', 5.0]), # Note: unary minus might need special handling later depending on evaluator
        ("100/0.5", [100.0, '/', 0.5]),
        (" ( 1 + 2 ) * 3 ", ['(', 1.0, '+', 2.0, ')', '*', 3.0]),
    ]

    for expr_str, expected_tokens in test_expressions_parse:
        try:
            # First, ensure it's valid (though parse_expression assumes valid input)
            if sanitize_input(expr_str):
                tokens = parse_expression(expr_str)
                print(f"Parse Expression: '{expr_str}' -> Tokens: {tokens}")
                if tokens != expected_tokens:
                    print(f"    Mismatch! Expected: {expected_tokens}, Got: {tokens}")
        except ValueError as e:
            # This might happen if sanitize_input fails, or parse_expression itself has an issue
            print(f"Parse Expression: '{expr_str}' -> Error: {e}")

    print("\nTesting parse_expression specific cases:")
    try:
        tokens = parse_expression(" ( 10.5 + 3 ) * -2 ")
        # Current simple tokenizer will make '-' and '2.0' separate tokens.
        # A more advanced parser/evaluator would handle unary minus.
        # For now, this is the expected output of this tokenizer.
        expected = ['(', 10.5, '+', 3.0, ')', '*', '-', 2.0]
        print(f"Parse ' ( 10.5 + 3 ) * -2 ' -> {tokens}")
        if tokens != expected:
            print(f"    Mismatch! Expected: {expected}, Got: {tokens}")
    except ValueError as e:
        print(f"Parse ' ( 10.5 + 3 ) * -2 ' -> Error: {e}")

    try:
        # This should ideally be caught by sanitize_input first,
        # but if parse_expression gets it, it might raise an error or misinterpret.
        # The current parse_expression might fail if sanitize_input didn't catch it.
        # sanitize_input should catch this.
        sanitize_input("1++2") # Test sanitize for this
        tokens = parse_expression("1++2")
        # Expected: [1.0, '+', '+', 2.0] by current regex
        # sanitize_input should catch "1++2" if it's considered invalid.
        # If sanitize_input allows it, parse_expression gives [1.0, '+', '+', 2.0]
        # _evaluate_parsed_tokens would then likely fail due to operator sequence.
        # Let's test this path with safe_eval_expression.
        print(f"Parse '1++2' -> {tokens}") # This will be [1.0, '+', '+', 2.0]
    except ValueError as e: # This error would come from sanitize_input if it disallowed "++"
        print(f"Parse '1++2' -> Error: {e}")


    # Example Usage for safe_eval_expression
    print("\n--- Testing safe_eval_expression ---")
    test_eval_expressions = [
        ("1 + 2", 3.0),
        (" (3.14 * 2) - 7 / (4 + 1) ", (3.14 * 2) - 7 / (4 + 1)), # 6.28 - 7 / 5 = 6.28 - 1.4 = 4.88
        ("12345", 12345.0),
        ("-5.0", -5.0),
        ("100 / 0.5", 200.0),
        (" ( 1 + 2 ) * 3 ", 9.0),
        ("2 * (3 + 4)", 14.0),
        ("10 - 4 * 2", 2.0), # 10 - 8
        ("50 / 2 * 5", 125.0), # (50/2)*5 = 25*5
        ("1 + -2", -1.0), # 1 + (0-2)
        ("-3 * -2", 6.0), # (0-3) * (0-2)
        ("10 / - (1 + 1)", -5.0), # 10 / -(2) = 10 / (0-2)
        ("2.5 * 4 - 1.5", 8.5), # 10 - 1.5
        ("1 + (2 * (3 - 1)) / 4", 2.0), # 1 + (2*2)/4 = 1 + 4/4 = 1+1
    ]

    for expr_str, expected_result in test_eval_expressions:
        try:
            result = safe_eval_expression(expr_str)
            print(f"Eval: '{expr_str}' -> Result: {result}, Expected: {expected_result}")
            if abs(result - expected_result) > 1e-9: # Compare floats with tolerance
                print(f"    MISMATCH! Got {result}, Expected {expected_result}")
        except (ValueError, ZeroDivisionError) as e:
            print(f"Eval: '{expr_str}' -> Error: {e} (Expected: {expected_result})")

    print("\n--- Testing safe_eval_expression error cases ---")
    error_expressions = [
        "1 + 2$", # Sanitize error
        "abc",    # Sanitize error
        "1 + (2 * 3!", # Sanitize error
        "eval('os.system(\"reboot\")')", # Sanitize error
        "1 / 0", # ZeroDivisionError
        "(1 + 2", # Mismatched parentheses
        "1 + * 2", # Parse or Eval error (sanitize might allow, parse might give weird tokens, eval should fail)
        "", # Empty expression
        "()", # Empty parentheses
        "1 ++ 2", # Sanitize might allow, parse gives [1,+,+,2], eval should fail
        "5 * ", # Incomplete expression
        "* 2", # Operator at start (not unary minus)
    ]

    for expr_str in error_expressions:
        try:
            result = safe_eval_expression(expr_str)
            print(f"Eval (error case): '{expr_str}' -> Result: {result} (UNEXPECTED SUCCESS)")
        except (ValueError, ZeroDivisionError) as e:
            print(f"Eval (error case): '{expr_str}' -> Expected Error: {e}")
        except Exception as e:
            print(f"Eval (error case): '{expr_str}' -> UNEXPECTED Exception type: {type(e).__name__} - {e}")
