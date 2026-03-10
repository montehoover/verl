import re

# Compile regex patterns once at module level for efficiency.
# These patterns define the allowed components of an expression.

# Numbers: integers (e.g., 123) and floats (e.g., 123.45).
# \b ensures matching whole words/numbers, preventing partial matches within identifiers.
_NUMBER_PATTERN = r'\b\d+(?:\.\d+)?\b'

# Keywords: logical operators (and, or, not) and boolean/None literals.
# \b ensures matching whole words.
_KEYWORD_PATTERN = r'\b(?:and|or|not|True|False|None)\b'

# Operators:
# Includes arithmetic (+, -, *, /, %, **, //),
# comparison (==, !=, <, >, <=, >=),
# and parentheses ( (, ) ).
# Order matters for multi-character operators (e.g., ** before *) to ensure correct matching.
_OPERATOR_PATTERN = r'\*\*|//|==|!=|<=|>=|[+\-*/%<>()]'

# A single token can be a number, a keyword, or an operator.
_TOKEN_PATTERN = rf'(?:{_NUMBER_PATTERN}|{_KEYWORD_PATTERN}|{_OPERATOR_PATTERN})'

# An expression is defined as a sequence of one or more blocks.
# Each block consists of:
#   1. Optional leading whitespace.
#   2. A valid token (number, keyword, or operator).
#   3. Optional trailing whitespace.
# The ^ and $ anchors ensure that the entire string conforms to this structure.
_VALID_EXPRESSION_PATTERN = re.compile(
    rf'^(\s*(?:{_TOKEN_PATTERN})\s*)+$'
)

def validate_expression(expression: str) -> bool:
    """
    Validates if the input expression string contains only allowed elements
    such as numbers, arithmetic/logical operators, comparison operators,
    parentheses, and specific keywords (and, or, not, True, False, None).
    It aims to filter out expressions with disallowed characters or constructs
    (e.g., variable names, function calls, assignments).

    Allowed elements:
    - Numbers (integers and floats, e.g., 123, 45.67)
    - Arithmetic operators: +, -, *, /, %, **, //
    - Logical operators (keywords): and, or, not
    - Comparison operators: ==, !=, <, >, <=, >=
    - Parentheses: (, )
    - Boolean literals: True, False
    - None literal: None
    - Whitespace

    Args:
        expression: The string expression to validate.

    Returns:
        True if the expression contains only allowed elements, False otherwise.

    Note:
    This function checks for the presence of valid tokens only. It does not
    guarantee that the expression is syntactically correct or semantically
    meaningful (e.g., "1 + * 2" or "1 / 0" would pass this validation as
    all components are individually valid tokens).
    """
    if not isinstance(expression, str):
        # Non-string inputs are considered invalid.
        return False

    # Reject empty strings or strings containing only whitespace,
    # as they don't form a meaningful expression.
    if not expression.strip():
        return False

    # Use re.fullmatch to ensure the entire string conforms to the defined pattern.
    if _VALID_EXPRESSION_PATTERN.fullmatch(expression):
        return True
    else:
        return False

if __name__ == '__main__':
    # Example Usage and Test Cases
    test_expressions = {
        "1 + 2": True,
        "10 * (2 - 3.5) / 4": True,
        "True and False": True,
        "not True": True,
        "1 == 1.0": True,
        "10 > 5 and (2 < 3 or 5 != 5)": True,
        "10 ** 2": True,
        "10 // 3": True,
        "10 % 3": True,
        "  1 + 2  ": True, # Whitespace handling
        "(1 + (2 * (3 / (4 % 5))))": True, # Nested parentheses
        "None is None": False, # "is" is not an allowed operator keyword
        "1.2.3": False, # Invalid number format
        "my_var + 1": False, # Disallowed identifier "my_var"
        "print('hello')": False, # Disallowed function call
        "1 + ; 2": False, # Disallowed character ";"
        "": False, # Empty string
        "   ": False, # Whitespace only string
        "1 + * 2": True, # Contains only valid tokens, though syntactically dubious
        "import os": False, # Disallowed keyword "import" and identifier "os"
        "1e5": False, # Scientific notation for numbers not currently supported by _NUMBER_PATTERN
        "0.5": True, # Standard float
        ".5": False, # Floats starting with "." not supported by _NUMBER_PATTERN (requires 0.5)
        "5.": False, # Floats ending with "." not supported by _NUMBER_PATTERN (requires 5.0 or 5)
    }

    for expr, expected in test_expressions.items():
        result = validate_expression(expr)
        print(f"Expression: \"{expr}\"")
        print(f"Expected: {expected}, Got: {result} -> {'PASS' if result == expected else 'FAIL'}")
        print("-" * 20)
