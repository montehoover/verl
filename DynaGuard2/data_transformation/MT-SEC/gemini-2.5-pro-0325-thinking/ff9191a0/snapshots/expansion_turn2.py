import re
from typing import List, Dict, Union

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

def substitute_variables(tokens: List[str], variables: Dict[str, Union[int, float]]) -> List[str]:
    """
    Substitutes variables in a list of tokens with their corresponding values.

    Args:
        tokens: A list of tokens (strings).
        variables: A dictionary mapping variable names to their numeric values.

    Returns:
        A new list of tokens with variables replaced by their stringified values.
    """
    substituted_tokens = []
    for token in tokens:
        if token in variables:
            substituted_tokens.append(str(variables[token]))
        else:
            substituted_tokens.append(token)
    return substituted_tokens

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

    # Example Usage for substitute_variables
    print("\n--- Variable Substitution Examples ---")
    expr_sub = "x * y + z / 2"
    tokens_sub = parse_expression(expr_sub)
    print(f"Original Expression: '{expr_sub}'")
    print(f"Original Tokens: {tokens_sub}")
    
    var_values = {"x": 5, "y": 10, "z": 4}
    substituted_tokens_sub = substitute_variables(tokens_sub, var_values)
    print(f"Variable Values: {var_values}")
    print(f"Substituted Tokens: {substituted_tokens_sub}")

    expr_sub2 = "radius * pi * radius"
    tokens_sub2 = parse_expression(expr_sub2)
    print(f"\nOriginal Expression: '{expr_sub2}'")
    print(f"Original Tokens: {tokens_sub2}")

    var_values2 = {"radius": 2.5, "pi": 3.14159}
    substituted_tokens_sub2 = substitute_variables(tokens_sub2, var_values2)
    print(f"Variable Values: {var_values2}")
    print(f"Substituted Tokens: {substituted_tokens_sub2}")

    # Example with a variable not in the dictionary
    expr_sub3 = "a + b * c"
    tokens_sub3 = parse_expression(expr_sub3)
    print(f"\nOriginal Expression: '{expr_sub3}'")
    print(f"Original Tokens: {tokens_sub3}")
    var_values3 = {"a": 1, "b": 2} # 'c' is missing
    substituted_tokens_sub3 = substitute_variables(tokens_sub3, var_values3)
    print(f"Variable Values: {var_values3}")
    print(f"Substituted Tokens (c not replaced): {substituted_tokens_sub3}")
