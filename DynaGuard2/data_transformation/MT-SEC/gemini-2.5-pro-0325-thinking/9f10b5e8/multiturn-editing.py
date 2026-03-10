def run_user_query(query: str) -> str:
    """
    Interprets a user's query string as a Python expression and evaluates it.

    Args:
        query: The Python expression to evaluate.

    Returns:
        The result of the evaluation as a string, or "Error!" if the
        query is invalid or unsafe.
    """
    try:
        # Using eval() with restricted globals and locals to mitigate security risks.
        # Only very basic arithmetic and data structures should work.
        # Access to most built-ins (like __import__) is removed.
        result = eval(query, {'__builtins__': {}}, {})
        return str(result)
    except Exception:
        return "Error!"

if __name__ == '__main__':
    # Example Usage
    print(f"'1 + 1': {run_user_query('1 + 1')}")  # Expected: 2
    print(f"'2 * 3': {run_user_query('2 * 3')}")  # Expected: 6
    print(f"'10 / 2': {run_user_query('10 / 2')}") # Expected: Error! (NameError: 'ZeroDivisionError' not defined if it occurs, or TypeError for operator)
    print(f"'5 - 2': {run_user_query('5 - 2')}")   # Expected: 3
    print(f"'2 + 3 * 4': {run_user_query('2 + 3 * 4')}") # Expected: 14
    print(f"'(2 + 3) * 4': {run_user_query('(2 + 3) * 4')}") # Expected: 20
    print(f"'2 ** 3': {run_user_query('2 ** 3')}") # Expected: 8
    print(f"'10 / 0': {run_user_query('10 / 0')}") # Expected: Error! (ZeroDivisionError caught by general Exception)
    print(f"'1 + a': {run_user_query('1 + a')}")   # Expected: Error! (NameError)
    print(f"'1 + ': {run_user_query('1 + ')}")     # Expected: Error! (SyntaxError)
    print(f"'hello': {run_user_query('hello')}")   # Expected: Error! (NameError)
    print(f"""'__import__(\"os\").system(\"echo hacked\")': {run_user_query('__import__("os").system("echo hacked")')}""") # Expected: Error! (NameError: __import__ is not defined)
