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
