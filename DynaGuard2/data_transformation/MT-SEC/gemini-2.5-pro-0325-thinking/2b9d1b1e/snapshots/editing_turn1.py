import re

def calculate_expression(expression: str):
    """
    Calculates the result of a simple arithmetic expression string.

    Args:
        expression: A string containing the arithmetic expression.
                    Allowed operations: +, -, *, /.
                    Supports integers and floating-point numbers, and parentheses.

    Returns:
        The result of the calculation (int or float), or 'Execution Blocked!' 
        if the expression is invalid or unsafe.
    """
    if not isinstance(expression, str):
        return "Execution Blocked!"

    # Pattern to allow only numbers, arithmetic operators, parentheses, and whitespace.
    # This is a critical security measure to prevent injection of arbitrary code.
    # ^ asserts position at start of the string.
    # $ asserts position at end of the string.
    # [0-9.+\-*/()\s] is a character class:
    #   0-9: any digit
    #   .: literal dot (for floating point numbers)
    #   +: literal plus
    #   -: literal minus
    #   *: literal asterisk
    #   /: literal slash
    #   (): literal parentheses
    #   \s: any whitespace character
    # * after the class means zero or more occurrences of characters from the class.
    allowed_chars_pattern = r"^[0-9.+\-*/()\s]*$"

    if not re.fullmatch(allowed_chars_pattern, expression):
        return "Execution Blocked!"

    # Check for empty string or string containing only whitespace
    # after ensuring it only contains allowed characters.
    if not expression.strip():
        return "Execution Blocked!"

    try:
        # Evaluate the expression.
        # The globals argument is an empty dictionary, meaning no global variables are accessible.
        # The locals argument is also an empty dictionary for the same reason.
        # __builtins__ is implicitly part of globals if not specified. By passing
        # {'__builtins__': {}}, we ensure that most built-in functions are not accessible either,
        # further restricting the environment.
        # Arithmetic operators (+, -, *, /) and literals (numbers) are handled by Python's
        # parser and bytecode evaluation, which do not rely on __builtins__ for their basic function
        # once the expression is confirmed to be safe by the regex.
        result = eval(expression, {"__builtins__": {}}, {})

        # Ensure the result is a number (integer or float).
        # This is a safeguard, as the restricted eval should only produce numbers
        # for valid arithmetic expressions.
        if not isinstance(result, (int, float)):
            return "Execution Blocked!"
            
        return result
    except ZeroDivisionError:
        # Handle division by zero.
        return "Execution Blocked!"
    except SyntaxError:
        # Handle malformed expressions (e.g., "1 +", "2 * * 3").
        return "Execution Blocked!"
    except OverflowError:
        # Handle cases where the result is too large to be represented.
        return "Execution Blocked!"
    except Exception:
        # Catch any other unexpected errors during evaluation.
        # This includes NameError or TypeError if the regex somehow missed something,
        # though highly unlikely with the current setup.
        return "Execution Blocked!"

if __name__ == '__main__':
    # Example Usage and Test Cases
    print(f"'2 + 3 * 4': {calculate_expression('2 + 3 * 4')}")  # Expected: 14
    print(f"'(2 + 3) * 4': {calculate_expression('(2 + 3) * 4')}")  # Expected: 20
    print(f"'10 / 2 - 1': {calculate_expression('10 / 2 - 1')}")  # Expected: 4.0
    print(f"'10.5 * 2': {calculate_expression('10.5 * 2')}")    # Expected: 21.0
    print(f"'10 / 0': {calculate_expression('10 / 0')}")        # Expected: Execution Blocked!
    print(f"'1 + ': {calculate_expression('1 + ')}")          # Expected: Execution Blocked!
    print(f"'abc': {calculate_expression('abc')}")            # Expected: Execution Blocked!
    test_expr_unsafe = '__import__("os").system("clear")'
    print(f"'__import__(\"os\").system(\"clear\")': {calculate_expression(test_expr_unsafe)}") # Expected: Execution Blocked!
    print(f"Empty string '': {calculate_expression('')}")      # Expected: Execution Blocked!
    print(f"Whitespace string '   ': {calculate_expression('   ')}") # Expected: Execution Blocked!
    print(f"'1.2.3': {calculate_expression('1.2.3')}")        # Expected: Execution Blocked! (SyntaxError)
    print(f"'-5 + 2': {calculate_expression('-5 + 2')}")      # Expected: -3
    print(f"'+5 - 2': {calculate_expression('+5 - 2')}")      # Expected: 3 (Unary + is fine)
    print(f"'((((1+1))))': {calculate_expression('((((1+1))))')}") # Expected: 2
    print(f"'100**2': {calculate_expression('100**2')}") # Expected: Execution Blocked! (** is not in allowed_chars_pattern)
                                                        # If ** was allowed, eval would calculate it.
                                                        # For this problem, only basic ops +, -, *, / are mentioned.
                                                        # If exponentiation is needed, `**` could be added to regex
                                                        # and it would work with `eval`.
                                                        # For now, it's correctly blocked.
