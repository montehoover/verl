def calculate_expression(expression: str) -> str:
    """
    Evaluates a simple arithmetic expression string and returns the result as a string.

    Args:
        expression: The arithmetic expression to evaluate (e.g., "2 + 3 * 4").

    Returns:
        The result of the calculation as a string, or "Error!" if the
        expression is invalid.
    """
    try:
        # Using eval() for simplicity. Be cautious with eval() in production
        # environments due to security risks if the input string is not trusted.
        result = eval(expression)
        return str(result)
    except Exception:
        return "Error!"

if __name__ == '__main__':
    # Example Usage
    print(f"'1 + 1': {calculate_expression('1 + 1')}")  # Expected: 2
    print(f"'2 * 3': {calculate_expression('2 * 3')}")  # Expected: 6
    print(f"'10 / 2': {calculate_expression('10 / 2')}") # Expected: 5.0
    print(f"'5 - 2': {calculate_expression('5 - 2')}")   # Expected: 3
    print(f"'2 + 3 * 4': {calculate_expression('2 + 3 * 4')}") # Expected: 14
    print(f"'(2 + 3) * 4': {calculate_expression('(2 + 3) * 4')}") # Expected: 20
    print(f"'10 / 0': {calculate_expression('10 / 0')}") # Expected: Error! (ZeroDivisionError)
    print(f"'1 + a': {calculate_expression('1 + a')}")   # Expected: Error! (NameError)
    print(f"'1 + ': {calculate_expression('1 + ')}")     # Expected: Error! (SyntaxError)
    print(f"'hello': {calculate_expression('hello')}")   # Expected: Error!
    print(f"""'__import__(\"os\").system(\"echo hacked\")': {calculate_expression('__import__("os").system("echo hacked")')}""") # Example of eval risk, should be Error! if restricted
