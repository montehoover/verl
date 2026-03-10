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

# Define a restricted environment for eval().
# Only True, False, and None are explicitly made available.
# Python's built-in keywords like 'and', 'or', 'not' are handled by the parser.
_EVAL_GLOBALS = {
    "__builtins__": {},
    "True": True,
    "False": False,
    "None": None,
}

def evaluate_valid_expression(expression: str) -> str:
    """
    Evaluates a validated arithmetic or logical expression string.

    The expression is first validated using `validate_expression`.
    If invalid, an error message regarding unauthorized elements is returned.
    If valid, the expression is evaluated in a restricted environment.
    Returns the computation result as a string, or an error message
    if evaluation fails (e.g., syntax error, division by zero).

    Args:
        expression: The string expression to evaluate.

    Returns:
        The result of the computation as a string, or an error message.
    """
    if not validate_expression(expression):
        return "Evaluation failed due to safety concerns: Expression contains unauthorized elements."

    try:
        # Evaluate the expression in a restricted environment.
        # The locals dictionary is empty.
        result = eval(expression, _EVAL_GLOBALS, {})
        return str(result)
    except SyntaxError:
        return "Evaluation failed: Malformed expression."
    except ZeroDivisionError:
        return "Evaluation failed: Division by zero."
    except TypeError:
        return "Evaluation failed: Type error in expression."
    except OverflowError:
        return "Evaluation failed: Numerical result out of range."
    except Exception as e:
        # Catch any other unexpected errors during evaluation.
        # It's good practice to log 'e' here in a real application.
        return f"Evaluation failed: An unexpected error occurred during computation."

def evaluate_user_math(math_expr: str) -> str:
    """
    Evaluates a user-submitted mathematical or logical expression string.

    This function acts as a wrapper around `validate_expression` and
    `evaluate_valid_expression` to provide a simplified interface for
    processing user input with a generic error message for any failure.

    Args:
        math_expr: The string expression to evaluate.

    Returns:
        The result of the computation as a string if successful,
        or "Processing Failed!" if any validation or evaluation error occurs.
    """
    # First, validate the expression for allowed tokens.
    # This is a preliminary check for obviously unsafe or malformed inputs.
    if not validate_expression(math_expr):
        return "Processing Failed!"

    # If validation passes, attempt to evaluate the expression.
    # evaluate_valid_expression handles its own specific errors (SyntaxError, ZeroDivisionError, etc.)
    # and returns a descriptive error message starting with "Evaluation failed:".
    result = evaluate_valid_expression(math_expr)

    # Check if the evaluation itself indicated a failure.
    if result.startswith("Evaluation failed"):
        return "Processing Failed!"

    # If no "Evaluation failed" message, then the result is considered successful.
    return result

if __name__ == '__main__':
    # Example Usage and Test Cases for validate_expression
    print("--- Testing validate_expression ---")
    test_expressions_validation = {
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

    for expr, expected in test_expressions_validation.items():
        result = validate_expression(expr)
        print(f"Validating: \"{expr}\"")
        print(f"Expected: {expected}, Got: {result} -> {'PASS' if result == expected else 'FAIL'}")
        print("-" * 20)

    print("\n--- Testing evaluate_valid_expression ---")
    test_expressions_evaluation = {
        # Valid expressions
        "1 + 2": "3",
        "10 * (2 - 3.5) / 4": "-3.75",
        "True and False": "False",
        "not True": "False",
        "1 == 1.0": "True",
        "10 > 5 and (2 < 3 or 5 != 5)": "True",
        "10 ** 2": "100",
        "10 // 3": "3",
        "10 % 3": "1",
        "  1 + 2  ": "3",
        "(1 + (2 * (3 / (4 % 5))))": "2.5", # 4 % 5 = 4; 3/4 = 0.75; 2*0.75=1.5; 1+1.5=2.5
        "0.5 * 2": "1.0",
        # Expressions that are token-valid but syntactically incorrect for eval
        "1 + * 2": "Evaluation failed: Malformed expression.",
        # Expressions leading to runtime errors
        "1 / 0": "Evaluation failed: Division by zero.",
        "10 / (1 - 1)": "Evaluation failed: Division by zero.",
        # Type errors (though Python bools can coerce to int for some ops)
        "True / 2": "0.5", # True becomes 1
        # "None + 1": "Evaluation failed: Type error in expression.", # This would be a TypeError
        # Expressions that should fail validation (and thus evaluation)
        "my_var + 1": "Evaluation failed due to safety concerns: Expression contains unauthorized elements.",
        "print('hello')": "Evaluation failed due to safety concerns: Expression contains unauthorized elements.",
        "1 + ; 2": "Evaluation failed due to safety concerns: Expression contains unauthorized elements.",
        "": "Evaluation failed due to safety concerns: Expression contains unauthorized elements.",
        "   ": "Evaluation failed due to safety concerns: Expression contains unauthorized elements.",
        "1e5": "Evaluation failed due to safety concerns: Expression contains unauthorized elements.",
        # Test with None (None itself is a valid token)
        "None": "None",
        # More complex boolean logic
        "(True or False) and not (False)": "True",
    }

    # Add a specific test case for TypeError that might occur with None
    # if not handled by Python's operator overloading.
    # For example, `None + 1` would raise a TypeError.
    # `validate_expression` allows `None`, `+`, `1`.
    # `eval` would then try `None + 1`.
    test_expressions_evaluation["None + 1"] = "Evaluation failed: Type error in expression."


    for expr, expected_str in test_expressions_evaluation.items():
        result_str = evaluate_valid_expression(expr)
        print(f"Evaluating (direct): \"{expr}\"")
        print(f"Expected: \"{expected_str}\", Got: \"{result_str}\" -> {'PASS' if result_str == expected_str else 'FAIL'}")
        print("-" * 20)

    print("\n--- Testing evaluate_user_math ---")
    test_expressions_user_math = {
        # Valid expressions that should succeed
        "1 + 2": "3",
        "10 * (2 - 3.5) / 4": "-3.75",
        "True and False": "False",
        "  (100 / 2) + 5  ": "55.0",
        "None": "None",
        "(True or False) and not (False)": "True",

        # Expressions that are token-valid but syntactically incorrect for eval
        "1 + * 2": "Processing Failed!",

        # Expressions leading to runtime errors
        "1 / 0": "Processing Failed!",
        "10 / (1 - 1)": "Processing Failed!",
        "None + 1": "Processing Failed!", # This causes a TypeError in evaluate_valid_expression

        # Expressions that should fail validation (and thus evaluation)
        "my_var + 1": "Processing Failed!",
        "print('hello')": "Processing Failed!",
        "1 + ; 2": "Processing Failed!",
        "": "Processing Failed!",
        "   ": "Processing Failed!",
        "1e5": "Processing Failed!", # Fails validate_expression
        "import os": "Processing Failed!", # Fails validate_expression
    }

    for expr, expected_output in test_expressions_user_math.items():
        user_result = evaluate_user_math(expr)
        print(f"Evaluating (user_math): \"{expr}\"")
        print(f"Expected: \"{expected_output}\", Got: \"{user_result}\" -> {'PASS' if user_result == expected_output else 'FAIL'}")
        print("-" * 20)
