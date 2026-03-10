import re

def evaluate_expression(expression_string):
    """
    Evaluates simple arithmetic expressions (addition, subtraction, multiplication, division) given as strings.

    Args:
        expression_string: The arithmetic expression string.

    Returns:
        The result of the evaluation as a string, or 'Invalid Expression!'
        if the input is invalid.
    """
    # Allow only digits, +, -, *, /, and spaces to prevent arbitrary code execution
    # and ensure it's a simple arithmetic expression.
    if not re.match(r"^[0-9\s\+\-\*\/]+$", expression_string):
        return 'Invalid Expression!'

    try:
        # Replace multiple spaces with single space for cleaner eval
        cleaned_expression = " ".join(expression_string.split())
        
        # Check for invalid patterns like "--" or "++" or leading/trailing operators
        # that might not be caught by eval but are not typical simple expressions.
        if not cleaned_expression or \
           cleaned_expression.startswith('+') or \
           cleaned_expression.startswith('-') and len(cleaned_expression) > 1 and not cleaned_expression[1].isdigit() or \
           cleaned_expression.endswith('+') or \
           cleaned_expression.endswith('-'):
             # Allow single number like "-5" but not just "-"
            if cleaned_expression == "-" or cleaned_expression == "+":
                 return 'Invalid Expression!'
            # Check if it's just a negative number, which is valid
            is_just_negative_number = cleaned_expression.startswith('-') and \
                                      all(c.isdigit() for c in cleaned_expression[1:])
            if not is_just_negative_number and (cleaned_expression.startswith('+') or \
                cleaned_expression.startswith('-') or \
                cleaned_expression.endswith('+') or \
                cleaned_expression.endswith('-')):
                 return 'Invalid Expression!'


        # A simple check to ensure there are numbers present.
        # This also helps prevent eval('') or eval(' ')
        if not any(char.isdigit() for char in cleaned_expression):
            return 'Invalid Expression!'

        # Basic check for operators not surrounded by numbers or valid constructs
        # e.g. "1 + + 2" or "1 -- 2"
        # This is a bit tricky with eval, as eval can handle some of these.
        # For "simple" expressions, we might want to be stricter.
        # The regex above already limits characters, this is an additional safety.
        # A more robust parser would be better for complex rules.

        result = eval(cleaned_expression)
        return str(result)
    except (SyntaxError, TypeError, NameError, ZeroDivisionError, OverflowError):
        return 'Invalid Expression!'
    except Exception: # Catch any other unexpected errors during eval
        return 'Invalid Expression!'

if __name__ == '__main__':
    # Test cases
    print(f"'5 + 6': {evaluate_expression('5 + 6')}")  # Expected: 11
    print(f"'10 - 4': {evaluate_expression('10 - 4')}") # Expected: 6
    print(f"' 3 +  7 - 2 ': {evaluate_expression(' 3 +  7 - 2 ')}") # Expected: 8
    print(f"'5': {evaluate_expression('5')}") # Expected: 5
    print(f"'-5': {evaluate_expression('-5')}") # Expected: -5
    print(f"'5 +': {evaluate_expression('5 +')}") # Expected: Invalid Expression!
    print(f"'+ 5': {evaluate_expression('+ 5')}") # Expected: Invalid Expression! (though eval might handle it)
    print(f"'abc': {evaluate_expression('abc')}") # Expected: Invalid Expression!
    print(f"'5 * 6': {evaluate_expression('5 * 6')}") # Expected: 30
    print(f"'10 / 2': {evaluate_expression('10 / 2')}") # Expected: 5.0
    print(f"'7 * 3 + 2': {evaluate_expression('7 * 3 + 2')}") # Expected: 23
    print(f"'10 - 4 / 2': {evaluate_expression('10 - 4 / 2')}") # Expected: 8.0
    print(f"'10 / 0': {evaluate_expression('10 / 0')}") # Expected: Invalid Expression!
    print(f"'5 / 2 * 4': {evaluate_expression('5 / 2 * 4')}") # Expected: 10.0
    eval_test_str = '''eval("__import__('os').system('clear')")'''
    print(f"'{eval_test_str}': {evaluate_expression(eval_test_str)}") # Expected: Invalid Expression!
    print(f"'': {evaluate_expression('')}") # Expected: Invalid Expression!
    print(f"'   ': {evaluate_expression('   ')}") # Expected: Invalid Expression!
    print(f"'5 - - 3': {evaluate_expression('5 - - 3')}") # Expected: Invalid Expression! (or 8 if eval handles it, but we aim for simple)
    print(f"'5 + + 3': {evaluate_expression('5 + + 3')}") # Expected: Invalid Expression!
    print(f"'-': {evaluate_expression('-')}") # Expected: Invalid Expression!
    print(f"'+': {evaluate_expression('+')}") # Expected: Invalid Expression!
    print(f"'10 + 2 - 3 + 5': {evaluate_expression('10 + 2 - 3 + 5')}") # Expected: 14
    print(f"'100000000000000000000 + 1': {evaluate_expression('100000000000000000000 + 1')}") # Expected: 100000000000000000001
    print(f"'5 + foo': {evaluate_expression('5 + foo')}") # Expected: Invalid Expression!
