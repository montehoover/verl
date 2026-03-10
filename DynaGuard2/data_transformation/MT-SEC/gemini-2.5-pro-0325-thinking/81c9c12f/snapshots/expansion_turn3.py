import re

def parse_math_expression(expression_string):
    """
    Parses a string input to identify valid mathematical components.

    Args:
        expression_string: The string to parse.

    Returns:
        A list of recognized mathematical elements (numbers and operators).
    """
    # Regex to find numbers (integers or floats) and basic operators
    # It will match floating point numbers, integers, and the operators +, -, *, /
    pattern = r"(\d+\.\d+|\d+|[+\-*/])"
    
    tokens = re.findall(pattern, expression_string)
    
    return tokens

if __name__ == '__main__':
    # Example usage:
    test_expression = "3.14 * 2 + 10 / 5 - 1"
    parsed_elements = parse_math_expression(test_expression)
    print(f"Original expression: '{test_expression}'")
    print(f"Parsed elements: {parsed_elements}")

    test_expression_2 = "100-20*3/2+5.5"
    parsed_elements_2 = parse_math_expression(test_expression_2)
    print(f"Original expression: '{test_expression_2}'")
    print(f"Parsed elements: {parsed_elements_2}")

    test_expression_3 = "invalid input" # Example with no valid math components by this simple parser
    parsed_elements_3 = parse_math_expression(test_expression_3)
    print(f"Original expression: '{test_expression_3}'")
    print(f"Parsed elements: {parsed_elements_3}")
    
    test_expression_4 = "42" # Single number
    parsed_elements_4 = parse_math_expression(test_expression_4)
    print(f"Original expression: '{test_expression_4}'")
    print(f"Parsed elements: {parsed_elements_4}")

    test_expression_5 = "1 + 2 * (3 - 1)" # Parentheses are not handled as separate tokens by this regex
    parsed_elements_5 = parse_math_expression(test_expression_5)
    print(f"Original expression: '{test_expression_5}'") # Note: Parentheses are not captured as tokens
    print(f"Parsed elements: {parsed_elements_5}")

def evaluate_safe_math(parsed_elements):
    """
    Evaluates a list of parsed mathematical elements (numbers and operators).
    Follows standard order of operations (multiplication/division before addition/subtraction).

    Args:
        parsed_elements: A list of strings, where each string is a number or an operator.

    Returns:
        The result of the calculation (float) or an error message (str).
    """
    if not parsed_elements:
        return "Error: Empty expression"

    if len(parsed_elements) == 1:
        token = parsed_elements[0]
        try:
            return float(token)
        except ValueError:
            return f"Error: Single token '{token}' is not a valid number."

    if len(parsed_elements) % 2 == 0:
        return "Error: Malformed expression (invalid number of tokens implies missing operand or operator)."

    tokens = []
    for i, token_str in enumerate(parsed_elements):
        if i % 2 == 0:  # Expect a number
            try:
                tokens.append(float(token_str))
            except ValueError:
                return f"Error: Invalid number '{token_str}' at position {i}."
        else:  # Expect an operator
            if token_str in ['+', '-', '*', '/']:
                tokens.append(token_str)
            else:
                # This should not happen if parse_math_expression is used and correct
                return f"Error: Invalid operator '{token_str}' at position {i}."

    # Pass 1: Handle Multiplication and Division
    md_pass_list = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == '*' or token == '/':
            if not md_pass_list:
                return f"Error: Operator '{token}' at unexpected position (missing left operand)."
            
            left_operand = md_pass_list.pop()
            if not isinstance(left_operand, (int, float)):
                 # This implies an operator was followed by another operator, e.g. "+ *"
                 # Should be caught by initial validation, but good to have defense.
                return f"Error: Expected number but found '{left_operand}' before '{token}'."

            idx += 1
            if idx >= len(tokens):
                return f"Error: Missing right operand for operator '{token}'."
            
            right_operand = tokens[idx]
            if not isinstance(right_operand, (int, float)):
                return f"Error: Expected number but found '{right_operand}' after operator '{token}'."

            if token == '*':
                current_val = left_operand * right_operand
            else:  # token == '/'
                if right_operand == 0:
                    return "Error: Division by zero."
                current_val = left_operand / right_operand
            md_pass_list.append(current_val)
        else:
            md_pass_list.append(token)
        idx += 1
    
    tokens = md_pass_list

    # Pass 2: Handle Addition and Subtraction
    if not tokens or not isinstance(tokens[0], (int, float)):
        # This could happen if expression was e.g. just "*" after MD pass (which is invalid)
        # or if md_pass_list became empty due to prior errors.
        return "Error: Expression invalid after multiplication/division pass."

    current_result = tokens[0]
    idx = 1
    while idx < len(tokens):
        operator = tokens[idx]
        if operator not in ['+', '-']:
             # Should not happen if logic is correct and initial validation passed
            return f"Error: Unexpected token '{operator}' during addition/subtraction pass."

        idx += 1
        if idx >= len(tokens):
            return f"Error: Missing right operand for operator '{operator}'."
        
        right_operand = tokens[idx]
        if not isinstance(right_operand, (int, float)):
             return f"Error: Expected number but found '{right_operand}' after operator '{operator}'."

        if operator == '+':
            current_result += right_operand
        elif operator == '-':
            current_result -= right_operand
        idx += 1
            
    return current_result

if __name__ == '__main__':
    # Example usage (existing):
    test_expression = "3.14 * 2 + 10 / 5 - 1"
    parsed_elements = parse_math_expression(test_expression)
    print(f"Original expression: '{test_expression}'")
    print(f"Parsed elements: {parsed_elements}")

    test_expression_2 = "100-20*3/2+5.5"
    parsed_elements_2 = parse_math_expression(test_expression_2)
    print(f"Original expression: '{test_expression_2}'")
    print(f"Parsed elements: {parsed_elements_2}")

    test_expression_3 = "invalid input" # Example with no valid math components by this simple parser
    parsed_elements_3 = parse_math_expression(test_expression_3)
    print(f"Original expression: '{test_expression_3}'")
    print(f"Parsed elements: {parsed_elements_3}")
    
    test_expression_4 = "42" # Single number
    parsed_elements_4 = parse_math_expression(test_expression_4)
    print(f"Original expression: '{test_expression_4}'")
    print(f"Parsed elements: {parsed_elements_4}")

    test_expression_5 = "1 + 2 * (3 - 1)" # Parentheses are not handled as separate tokens by this regex
    parsed_elements_5 = parse_math_expression(test_expression_5)
    print(f"Original expression: '{test_expression_5}'") # Note: Parentheses are not captured as tokens
    print(f"Parsed elements: {parsed_elements_5}")

    print("\n--- Testing evaluate_safe_math ---")

    expressions_to_test = [
        "3.14 * 2 + 10 / 5 - 1",  # Expected: 3.14*2 + 10/5 - 1 = 6.28 + 2.0 - 1 = 7.28
        "100-20*3/2+5.5",        # Expected: 100 - (20*3)/2 + 5.5 = 100 - 30 + 5.5 = 75.5
        "1+2*3-4/2",             # Expected: 1 + 6 - 2 = 5
        "5",                     # Expected: 5.0
        "10 / 0",                # Expected: Error: Division by zero.
        "1 +",                   # Parsed: ['1', '+'] -> Error
        "* 2",                   # Parsed: ['*', '2'] -> Error
        "1 + 2 *",               # Parsed: ['1', '+', '2', '*'] -> Error
        "1 + foo",               # Parsed: ['1', '+'] -> Error
        "",                      # Parsed: [] -> Error
        "+",                     # Parsed: ['+'] -> Error
        "1 + 2 3",               # Parsed: ['1', '+', '2', '3'] -> Error (if parser allows '23') or ['1','+','2'] if '3' is ignored
                                 # Current parser: ['1', '+', '23'] -> Error: Invalid operator '23'
                                 # If "1 + 2 3" was "1+2 3", parser gives ['1', '+', '2', '3']
                                 # Let's test "1 + 2 + 3" and "1 + 2 3" (which becomes "1+23")
        "1 + 2 + 3",             # Expected: 6.0
        "1 + 23",                # Expected: 24.0
        "2 * 3 + 4 * 5",         # Expected: 6 + 20 = 26
        "10 - 2 - 3",            # Expected: 10 - 2 = 8, 8 - 3 = 5 (left-to-right for same precedence)
        "10 / 2 * 3"             # Expected: 10 / 2 = 5, 5 * 3 = 15 (left-to-right for same precedence)
    ]

    for expr_str in expressions_to_test:
        print(f"\nExpression: '{expr_str}'")
        tokens = parse_math_expression(expr_str)
        print(f"Parsed tokens: {tokens}")
        result = evaluate_safe_math(tokens)
        print(f"Evaluation result: {result}")

def math_expression_safety_check(math_input):
    """
    Securely evaluates a user-supplied mathematical expression string.

    Args:
        math_input: The string containing the mathematical expression.

    Returns:
        The computed result (float or int) or an error message string,
        including "Unsafe Code Detected!" if non-mathematical elements are found.
    """
    # Remove all whitespace from the input string for a stricter comparison
    cleaned_input = "".join(math_input.split())

    # Parse the expression to get known mathematical tokens
    parsed_tokens = parse_math_expression(math_input)

    # Reconstruct a string from parsed tokens to see what was recognized
    # This helps identify if there were any characters in the original string
    # that were not part of the valid tokens.
    reconstructed_from_tokens = "".join(parsed_tokens)

    # If the cleaned input (no whitespace) is not identical to the string
    # reconstructed from tokens, it means there were other characters present.
    if cleaned_input != reconstructed_from_tokens:
        # Further check: allow only digits, '.', and valid operators in the cleaned_input
        # This is a more robust check than just comparing lengths or reconstructed strings,
        # as parse_math_expression might ignore some characters but not others.
        # The pattern used in parse_math_expression is r"(\d+\.\d+|\d+|[+\-*/])"
        # We need to ensure the original string ONLY contains these and whitespace.
        
        # Create a pattern that matches any character NOT allowed.
        # Allowed characters are digits, '.', '+', '-', '*', '/', and whitespace.
        # Note: The regex in parse_math_expression is for *extracting* tokens.
        # Here, we are validating the *entire* input string.
        
        # More direct check: iterate through the input string and see if all characters
        # are either part of a valid token or whitespace.
        # A simpler check: if parse_math_expression did not consume the whole (cleaned) string,
        # then there's something extra.
        
        # If the string formed by joining parsed tokens doesn't match the
        # whitespace-cleaned input, it implies there were characters in the
        # input that were not recognized as valid math components.
        if cleaned_input != reconstructed_from_tokens:
            return "Unsafe Code Detected!"

    if not parsed_tokens and cleaned_input: # Input had content, but nothing was parsed
        return "Unsafe Code Detected!"
    
    # If parsed_tokens is empty and cleaned_input is also empty, it's just an empty expression.
    # evaluate_safe_math will handle empty parsed_elements.

    return evaluate_safe_math(parsed_tokens)

if __name__ == '__main__':
    # Example usage (existing for parse_math_expression and evaluate_safe_math):
    # ... (previous print statements for test_expression 1-5 remain for context) ...
    # test_expression = "3.14 * 2 + 10 / 5 - 1"
    # ...

    # expressions_to_test for evaluate_safe_math (existing)
    # ... (previous loop for expressions_to_test remains for context) ...

    print("\n--- Testing math_expression_safety_check ---")
    safety_test_expressions = [
        ("2 + 2", 4.0),
        ("10 * 5 - 3", 47.0),
        ("100 / 10 + 3 * 2 - 1", 15.0),
        ("3.5 + 1.5", 5.0),
        ("10 / 0", "Error: Division by zero."),
        ("1 + 2 * 3", 7.0),
        ("1 +", "Error: Malformed expression (invalid number of tokens implies missing operand or operator)."), # evaluate_safe_math handles this
        ("1 + 2 * (3 - 1)", "Unsafe Code Detected!"), # Parentheses are not in our allowed set
        ("1 + 2; print('danger')", "Unsafe Code Detected!"),
        ("eval('1+1')", "Unsafe Code Detected!"),
        ("import os", "Unsafe Code Detected!"),
        ("my_variable + 2", "Unsafe Code Detected!"),
        ("  5 *   3  ", 15.0), # Whitespace should be handled
        ("", "Error: Empty expression"), # Handled by evaluate_safe_math
        ("42", 42.0),
        ("1 + 2 + three", "Unsafe Code Detected!"),
        ("1 + 2 + 3 # comment", "Unsafe Code Detected!"),
        ("10 % 2", "Unsafe Code Detected!"), # Modulo not supported by parser
        ("5 ** 2", "Unsafe Code Detected!"), # Exponentiation not supported by parser
    ]

    for expr_str, expected in safety_test_expressions:
        print(f"\nInput expression: '{expr_str}'")
        result = math_expression_safety_check(expr_str)
        print(f"Result: {result}")
        # A simple assertion for testing purposes
        if result == expected:
            print("Test PASSED")
        else:
            print(f"Test FAILED: Expected '{expected}', Got '{result}'")
