import re

def parse_expression(expression: str) -> list[str]:
    """
    Parses a mathematical expression string into a list of tokens.

    Tokens can be numbers (integers or decimals), operators (+, -, *, /, ^, (, )),
    or variable names (alphanumeric, starting with a letter).

    Args:
        expression: The mathematical expression string.

    Returns:
        A list of tokens.
    """
    # Pattern to match numbers (integers and decimals),
    # operators (+, -, *, /, ^, parentheses),
    # and variable names (alphanumeric, starting with a letter, allowing underscores)
    token_pattern = re.compile(r"""
        \s*                               # Optional whitespace
        (
            \d+\.\d* | \.\d+ | \d+ |    # Numbers (float or integer)
            [a-zA-Z_][a-zA-Z0-9_]* |    # Variable names
            [+\-*/^()]                  # Operators and parentheses
        )
        \s*                               # Optional whitespace
    """, re.VERBOSE)
    
    tokens = token_pattern.findall(expression)
    return tokens

if __name__ == '__main__':
    # Example Usage
    expr1 = "3 + 4 * (2 - 1) / x ^ 2"
    tokens1 = parse_expression(expr1)
    print(f"Expression: '{expr1}'")
    print(f"Tokens: {tokens1}")

    expr2 = "  var1*2.5 - (another_var / 3)  "
    tokens2 = parse_expression(expr2)
    print(f"Expression: '{expr2}'")
    print(f"Tokens: {tokens2}")

    expr3 = "sin(x) + cos(y)" # Note: 'sin' and 'cos' will be treated as variables here
    tokens3 = parse_expression(expr3)
    print(f"Expression: '{expr3}'")
    print(f"Tokens: {tokens3}")
    
    expr4 = "a+b-c*d/e^f"
    tokens4 = parse_expression(expr4)
    print(f"Expression: '{expr4}'")
    print(f"Tokens: {tokens4}")

    expr5 = "(1.0 + 2) * 3.14"
    tokens5 = parse_expression(expr5)
    print(f"Expression: '{expr5}'")
    print(f"Tokens: {tokens5}")
