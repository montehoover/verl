import re

def calculate_expression(expr_input):
    """
    Evaluates user-provided mathematical expressions in Python syntax.

    Args:
        expr_input: The mathematical expression string.

    Returns:
        The result of the evaluation as a string, or 'Computation Error!'
        if the input is invalid or suspicious.
    """
    # Allow only digits, operators (+, -, *, /), parentheses, and spaces.
    # This is a primary safety filter.
    if not re.match(r"^[0-9\s\+\-\*\/\(\)]+$", expr_input):
        return 'Computation Error!'

    try:
        # Replace multiple spaces with single space for cleaner eval and checks
        cleaned_expression = " ".join(expr_input.split())

        # Check 1: Empty string after cleaning
        if not cleaned_expression:
            return 'Computation Error!'

        # Check 2: Expression must contain at least one digit
        # This catches "()", "( )", "+ -", " ", etc.
        if not any(char.isdigit() for char in cleaned_expression):
            return 'Computation Error!'

        # Check 3: Invalid leading characters/operators
        first_char = cleaned_expression[0]
        if first_char in ('*', '/'):  # Cannot start with * or /
            return 'Computation Error!'
        # Cannot start with + or - unless it's part of a number or followed by a digit/paren
        if first_char in ('+', '-') and \
           (len(cleaned_expression) == 1 or \
            (not cleaned_expression[1].isdigit() and cleaned_expression[1] != '(')):
            return 'Computation Error!' # Catches "+", "-", "+ ", "- *", etc.

        # Check 4: Invalid trailing characters/operators
        # Cannot end with any operator or an opening parenthesis
        last_char = cleaned_expression[-1]
        if last_char in ('+', '-', '*', '/', '('):
            return 'Computation Error!'

        # If all checks pass, attempt to evaluate the expression
        result = eval(cleaned_expression)
        return str(result)
    except (SyntaxError, TypeError, NameError, ZeroDivisionError, OverflowError):
        return 'Computation Error!'
    except Exception:  # Catch any other unexpected errors during eval
        return 'Computation Error!'

if __name__ == '__main__':
    # Test cases
    print(f"'5 + 6': {calculate_expression('5 + 6')}")  # Expected: 11
    print(f"'10 - 4': {calculate_expression('10 - 4')}") # Expected: 6
    print(f"' 3 +  7 - 2 ': {calculate_expression(' 3 +  7 - 2 ')}") # Expected: 8
    print(f"'5': {calculate_expression('5')}") # Expected: 5
    print(f"'-5': {calculate_expression('-5')}") # Expected: -5
    print(f"'+5': {calculate_expression('+5')}") # Expected: 5 (eval handles unary plus)
    print(f"'5 +': {calculate_expression('5 +')}") # Expected: Computation Error!
    print(f"'+ 5': {calculate_expression('+ 5')}") # Expected: Computation Error! (stricter check)
    print(f"'- 5': {calculate_expression('- 5')}") # Expected: Computation Error! (stricter check)
    print(f"'* 5': {calculate_expression('* 5')}") # Expected: Computation Error!
    print(f"'/ 5': {calculate_expression('/ 5')}") # Expected: Computation Error!
    print(f"'5 *': {calculate_expression('5 *')}") # Expected: Computation Error!
    print(f"'abc': {calculate_expression('abc')}") # Expected: Computation Error! (regex)
    print(f"'5 * 6': {calculate_expression('5 * 6')}") # Expected: 30
    print(f"'10 / 2': {calculate_expression('10 / 2')}") # Expected: 5.0
    print(f"'7 * 3 + 2': {calculate_expression('7 * 3 + 2')}") # Expected: 23
    print(f"'10 - 4 / 2': {calculate_expression('10 - 4 / 2')}") # Expected: 8.0
    print(f"'(5 + 3) * 2': {calculate_expression('(5 + 3) * 2')}") # Expected: 16
    print(f"'10 / (2 - 2)': {calculate_expression('10 / (2 - 2)')}") # Expected: Computation Error! (ZeroDivisionError)
    print(f"'10 / 0': {calculate_expression('10 / 0')}") # Expected: Computation Error! (ZeroDivisionError)
    print(f"'5 / 2 * 4': {calculate_expression('5 / 2 * 4')}") # Expected: 10.0
    print(f"'(10)': {calculate_expression('(10)')}") # Expected: 10
    print(f"'(-5)': {calculate_expression('(-5)')}") # Expected: -5
    print(f"'(+5)': {calculate_expression('(+5)')}") # Expected: 5
    eval_test_str = '''eval("__import__('os').system('clear')")'''
    print(f"'{eval_test_str}': {calculate_expression(eval_test_str)}") # Expected: Computation Error! (regex)
    print(f"'': {calculate_expression('')}") # Expected: Computation Error!
    print(f"'   ': {calculate_expression('   ')}") # Expected: Computation Error!
    print(f"'()': {calculate_expression('()')}") # Expected: Computation Error! (no digits)
    print(f"'( )': {calculate_expression('( )')}") # Expected: Computation Error! (no digits)
    print(f"'+ -': {calculate_expression('+ -')}") # Expected: Computation Error! (no digits)
    print(f"'5 - - 3': {calculate_expression('5 - - 3')}") # Expected: 8 (valid Python)
    print(f"'5 + + 3': {calculate_expression('5 + + 3')}") # Expected: 8 (valid Python)
    print(f"'-': {calculate_expression('-')}") # Expected: Computation Error!
    print(f"'+': {calculate_expression('+')}") # Expected: Computation Error!
    print(f"'*': {calculate_expression('*')}") # Expected: Computation Error! (no digits)
    print(f"'/': {calculate_expression('/')}") # Expected: Computation Error! (no digits)
    print(f"'(' : {calculate_expression('(')}") # Expected: Computation Error!
    print(f"')' : {calculate_expression(')')}") # Expected: Computation Error!
    print(f"'10 + 2 - 3 + 5': {calculate_expression('10 + 2 - 3 + 5')}") # Expected: 14
    print(f"'100000000000000000000 + 1': {calculate_expression('100000000000000000000 + 1')}") # Expected: 100000000000000000001
    print(f"'5 + foo': {calculate_expression('5 + foo')}") # Expected: Computation Error! (regex)
    print(f"'5+3*2-(10/2)': {calculate_expression('5+3*2-(10/2)')}") # Expected: 6.0
    print(f"'((5+3)*2)/4': {calculate_expression('((5+3)*2)/4')}") # Expected: 4.0
    print(f"'1 + (2 * (3 - 1)) / 2': {calculate_expression('1 + (2 * (3 - 1)) / 2')}") # Expected: 3.0
